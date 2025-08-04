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

#EAFSC
class EAFSC(nn.Module):
    def __init__(self, args, layers, nhid, alpha=0.2):
        super(EAFSC, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        # self.nhid=nhid
        #self.mu determined by the init method
        # self.alpha=alpha
        self.CLmodel = AFSC(layers, args).to(self._device)

    def forward(self, data,image_data):
        data.to(self._device)
        feat_x = torch.tensor(data.x).float().to(self._device)
        x, _, loss = self.CLmodel(x=feat_x, y=data.y,image=image_data, edge_index=data.edge_index, 
                                              neighbor=[data.neighbor_index, data.neighbor_attr], 
                                              edge_weight=data.edge_attr)
        return x, loss    
    
    def fit(self, data,image_data,  lr=0.001, max_epochs=5000,update_interval=3, trajectory_interval=50,weight_decay=5e-4,opt="sgd"):
        
        if opt=="sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        for epoch in range(max_epochs):
            # print(epoch)
            optimizer.zero_grad()
            if epoch%update_interval == 0:
                  data.to(self._device)
                  x, CLloss = self.forward(data,image_data)
            
            x, CLloss = self.forward(data,image_data)
            loss = CLloss
            loss.backward()
            optimizer.step()
            self.CLmodel.update_moving_average()
            # print("epoch:{:.2f},loss:{:.5f}".format(epoch,loss))


    def predict(self,data,image_data,init="louvain",res=0.4,n_clusters=10,init_spa=True,tol=1e-5):
        x,_ = self(data,image_data)
        features_pca = pca(x.cpu().detach().numpy(),50)
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            kmeans = KMeans(n_clusters, init='k-means++', n_init=10)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features_pca)
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(data.x)  #Here we use X as numpy
        elif init=="louvain":
            if init_spa:
                adata=sc.AnnData(features_pca)
            else:
                adata=sc.AnnData(data.x)
            sc.pp.neighbors(adata)
            target_clusters = n_clusters  # 目标聚类数

            # 二分搜索法寻找合适分辨率
            low, high = 0.0, 3.0
            tolerance = 0  # 允许的聚类数偏差
            best_res = None

            for _ in range(50):  # 最多迭代20次
                mid = (low + high) / 2
                sc.tl.louvain(adata, resolution=mid, key_added=f"louvain_temp")
                n_clusters_ = adata.obs['louvain_temp'].nunique()
                
                # print(f"Resolution={mid:.3f} → {n_clusters_} clusters")
                
                if abs(n_clusters_ - target_clusters) <= tolerance:
                    best_res = mid
                    break
                elif n_clusters_ < target_clusters:
                    low = mid  # 聚类太少则增加分辨率
                else:
                    high = mid  # 聚类太多则降低分辨率

            # 应用最佳分辨率
            if best_res is not None:
                sc.tl.louvain(adata, resolution=best_res, key_added="louvain")
                print(f"Found resolution {best_res:.3f} for {target_clusters} clusters")
            else:
                print("未找到精确匹配，使用最后一次结果")
                adata.obs['louvain'] = adata.obs['louvain_temp']

            y_pred=adata.obs['louvain'].astype(int).to_numpy()
        return x,y_pred
    
# 聚类
class EAFSCcluster(object):
    def __init__(self):
        super(EAFSCcluster, self).__init__()


    def train(self,data,image_data,args, 
            num_pcs=50, 
            lr=0.001,
            max_epochs=2000,
            nhid=512,
            weight_decay=0,
            opt="admin",
            init_spa=True,
            init="kmeans", #louvain or kmeans
            n_clusters=4, #for kmeans
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
        self.n_clusters=n_clusters
        self.res=res
        self.tol=tol
        #----------Train model----------
        layers = [data.x.shape[1]]+eval(args.layers)
        self.model=EAFSC(args,layers,self.nhid)
        self.model.fit(data,image_data,lr=self.lr,max_epochs=self.max_epochs,weight_decay=self.weight_decay,opt=self.opt)
        self.data=data
        self.image_data=image_data

    def predict(self):
        x,y_pred=self.model.predict(self.data,self.image_data,init=self.init,n_clusters=self.n_clusters,res=self.res, tol=self.tol)
        return x,y_pred


