from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import copy
from sklearn.cluster import KMeans
from src import *
import faiss

#编码器

class GraphEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(GraphEncoder, self).__init__()
        self.gc_feat = GCNConv(input_dims, hidden_dims)
        self.gc_mean = GCNConv(hidden_dims, output_dims)
        self.gc_var = GCNConv(hidden_dims, output_dims)

    def forward(self, x, edge_index, edge_weight):
        x = self.gc_feat(x, edge_index, edge_weight).relu()
        mean = self.gc_mean(x, edge_index, edge_weight)
        var = self.gc_var(x, edge_index, edge_weight)
        return mean, var


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList([nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            # print(gnn._cached_edge_index,gnn.node_dim)
            x = gnn(x, edge_index,edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)
        return x

#AFSC
class AFSC(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        self.neighbor = Neighbor(args)

        rep_dim = layer_config[-1]
        # print(layer_config,rep_dim)

        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.PReLU(), nn.Linear(args.pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

        self.topk = args.topk
        self.image_prdictor = nn.Sequential(nn.Linear(1024, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.PReLU(), nn.Linear(args.pred_hid, args.pca_dim))
        # self.gene_prdictor = nn.Sequential(nn.Linear(args.pca_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.PReLU(), nn.Linear(args.pred_hid, 128))
        self.fusion = multimodal_fusion_layer(model_dim = args.pca_dim,num_heads=4, ffn_dim=args.pca_dim)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        print("init_model")

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
        
    def contrastive_loss(self,logits: torch.Tensor):
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self,similarity: torch.Tensor):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    def forward(self, x, y, image, edge_index, neighbor, edge_weight=None, epoch=None):

        image_data = self.image_prdictor(image)
        # gene_data = self.gene_prdictor(x)
        feat_fuse = self.fusion(x,image_data)
        # print(feat_fuse.shape)


        student = self.student_encoder(x=feat_fuse, edge_index=edge_index)
        
        pred = self.student_predictor(student)
        # print(image.device)
        
        
        # print(student.shape,image_data.shape)
        # print('fuse')
        
        # feat_fuse = torch.cat((student,image_data),dim=1)
        # feat_fuse = feat_fuse / feat_fuse.norm(dim=-1, keepdim=True)
        # feat_fuse = student+image_data
        # print(student.device)

        with torch.no_grad():
            teacher = self.teacher_encoder(x=feat_fuse, edge_index=edge_index, edge_weight=edge_weight)

        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]), [x.shape[0], x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [x.shape[0], x.shape[0]])
         
        ind, k = self.neighbor(adj.to(self._device), nn.functional.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk, epoch)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss3 = loss_fn(x,image_data)
        
        # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        # logits_per_text = torch.matmul(student, image_data.t()) * logit_scale

        # loss3 = self.clip_loss(logits_per_text)



        loss = (loss1 + loss2).mean()+loss3.mean()
        # loss = (loss1 + loss2).mean()
        # print(loss3.mean(),loss)
        
        

        return student, feat_fuse, loss

#正样本集
class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        self.device = args.device
        self.num_centroids = args.num_centroids
        self.num_kmeans = args.num_kmeans
        self.clus_num_iters = args.clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k, epoch):

        

        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj
        

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)
        
            clust_labels = I_kmeans[:,0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)
        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):
        
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])
        assert len(similar) == len(index)
        if type(index) is np.ndarray:
            indices = torch.tensor(np.array([index, similar])).to(self.device)
        else:
            indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    


    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result


#注意力
class multimodal_attention(torch.nn.Module):
    """
    dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = torch.nn.Dropout(attention_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))
        if scale:
            # print(attention,scale)
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        # print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))

        return attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = torch.nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = torch.nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = torch.nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = torch.nn.Linear(model_dim, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        #print("query.shape:{}".format(query.shape))

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        #batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        #print('key.shape:{}'.format(key.shape))

        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        # print(query, key, value,scale)
        attention = self.dot_product_attention(query, key, value, 
                                               scale, attn_mask)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        #print('attention_con_shape:{}'.format(attention.shape))

        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        #print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(torch.nn.Module):
    """
    Fully-connected network 
    """
    def __init__(self, model_dim=768, ffn_dim=768, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = torch.nn.Linear(model_dim, ffn_dim)
        self.w2 = torch.nn.Linear(ffn_dim, model_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(torch.nn.functional.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(torch.nn.Module):
    """
    A layer of fusing features 
    """
    def __init__(self, model_dim=768, num_heads=4, ffn_dim=768, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)
        
        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        
        self.fusion_linear = torch.nn.Linear(model_dim*2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):

        output_1 = self.attention_1(image_output, text_output, text_output,
                                 attn_mask)
        
        output_2 = self.attention_2(text_output, image_output, image_output,
                                 attn_mask)
        
        
        #print('attention out_shape:{}'.format(output.shape))
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)
        
        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output