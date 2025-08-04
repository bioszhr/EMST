from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import copy
from sklearn.cluster import KMeans
from src import *
import faiss
from torch.utils.data import DataLoader, Dataset

class Dataset(Dataset):
    def __init__(self, data, image_data, mu=None):
        super(Dataset, self).__init__()
        
        # adata, label, image
        self.gene = data
        self.image = image_data
        self.mu = mu

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        xg = self.gene[idx]
        xi = self.image[idx]
        if self.mu is not None:
            xmu = self.mu[idx]
            return xg, xi, xmu
        else:
            return xg, xi

#MAFSC
class MAFSC(nn.Module):
    def __init__(self, args, layers, nhid, alpha=0.2):
        super(MAFSC, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        self.nhid=nhid
        #self.mu determined by the init method
        self.alpha=alpha
        self.CLmodel = AFSC(layers, args).to(self._device)

    def forward(self, data,image_data):
        data.to(self._device)
        feat_x = torch.tensor(data.x).float().to(self._device)
        x, fuse, loss = self.CLmodel(x=feat_x, y=data.y,image=image_data, edge_index=data.edge_index, 
                                              neighbor=[data.neighbor_index, data.neighbor_attr], 
                                              edge_weight=data.edge_attr)
        fuse=fuse.cpu()
        q = 1.0 / ((1.0 + torch.sum((fuse.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, fuse, q, loss

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    
    
    def fit(self, data,image_data,  lr=0.001, max_epochs=5000,update_interval=3, trajectory_interval=50,weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-5):
        
        if opt=="sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        data.to(self._device)
        feat_x = torch.tensor(data.x).float().to(self._device)
        _, features, _ = self.CLmodel(x=feat_x, y=data.y,image=image_data, edge_index=data.edge_index, 
                                              neighbor=[data.neighbor_index, data.neighbor_attr], 
                                              edge_weight=data.edge_attr)
        features = features.cpu()
        # print(features)
        features_pca = pca(features.detach().numpy(),50)
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, init='k-means++', n_init=10)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features_pca)
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(data.x)  #Here we use X as numpy
        elif init=="louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(features_pca)
            else:
                adata=sc.AnnData(data.x)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.nhid))
        features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        print("train")
        for epoch in range(max_epochs):
          # print(epoch)
          optimizer.zero_grad()
          if epoch%update_interval == 0:
              _, _, q, CLloss = self.forward(data,image_data)
              p = self.target_distribution(q).data
          
          _, _, q, CLloss = self.forward(data,image_data)
          # loss = self.loss_function(p, q) + CLloss
          loss = self.loss_function(p, q) + CLloss
          # print(self.loss_function(p, q),loss)
          loss.backward()
          optimizer.step()
          self.CLmodel.update_moving_average()

          #Check stop criterion
          y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
          delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / data.x.shape[0]
          y_pred_last = y_pred
          if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
              print('delta_label ', delta_label, '< tol ', tol)
              print("Reach tolerance threshold. Stopping training.")
              print("Total epoch:", epoch)
              break


    def predict(self,data,image_data):
        z,z_fuse,q,_ = self(data,image_data)
        return z, z_fuse, q
    
# 聚类
class cluster(object):
    def __init__(self):
        super(cluster, self).__init__()


    def train(self,data,image_data,args, 
            num_pcs=50, 
            lr=0.001,
            max_epochs=2000,
            nhid=512,
            weight_decay=0,
            opt="admin",
            init_spa=True,
            init="kmeans", #louvain or kmeans
            n_neighbors=52, #for louvain
            n_clusters=52, #for kmeans
            res=0.7, #for louvain
            tol=1e-5):
        self.num_pcs=num_pcs
        self.res=res
        self.lr=lr
        self.max_epochs=max_epochs
        self.nhid=nhid
        self.weight_decay=weight_decay
        self.opt=opt
        self.init_spa=init_spa
        self.init=init
        self.n_neighbors=n_neighbors
        self.n_clusters=n_clusters
        self.res=res
        self.tol=tol
        #----------Train model----------
        layers = [data.x.shape[1]]+eval(args.layers)
        self.model=MAFSC(args,layers,self.nhid)
        self.model.fit(data,image_data,lr=self.lr,max_epochs=self.max_epochs,weight_decay=self.weight_decay,opt=self.opt,init_spa=self.init_spa,init=self.init,n_neighbors=self.n_neighbors,n_clusters=self.n_clusters,res=self.res, tol=self.tol)
        self.data=data
        self.image_data=image_data

    def predict(self):
        z,z_fuse,q=self.model.predict(self.data,self.image_data)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        return y_pred, z, z_fuse


