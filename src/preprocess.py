""" pca_lowrank """
import math
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import torch
from sklearn.neighbors import NearestNeighbors 
import torch_geometric
import os
import anndata
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def large_mat_mul(input_a, input_b, batch=32):
    m = input_a.shape[0]
    block_m = math.floor(m / batch)
    out = []
    for i in range(batch):
        start = i * block_m
        end = (i + 1) * block_m
        new_a = input_a[start:end]
        out_i = np.matmul(new_a, input_b)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    remain_a = input_a[batch * block_m:m]
    remain_o = np.matmul(remain_a, input_b)
    output = np.concatenate((out, remain_o), axis=0)
    return output


def mat_mul(input_a, input_b):
    m = input_a.shape[0]
    if m > 100000:
        out = large_mat_mul(input_a, input_b)
    else:
        out = np.matmul(input_a, input_b)

    return out


def get_approximate_basis(matrix: np.ndarray,
                          q=6,
                          niter=2,
                          ):
    niter = 2 if niter is None else niter
    _, n = matrix.shape[-2:]

    r = np.random.randn(n, q)

    matrix_t = matrix.T

    q, _ = np.linalg.qr(mat_mul(matrix, r))
    for _ in range(niter):
        q = np.linalg.qr(mat_mul(matrix_t, q))[0]
        q = np.linalg.qr(mat_mul(matrix, q))[0]
    return q


def pca(matrix: np.ndarray, k: int = None, niter: int = 2, norm: bool = False):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The matrix type is {},\
                        but it should be ndarray.".format(type(matrix)))
    if not isinstance(k, int):
        raise TypeError("The k type is {},\
                        but it should be int.".format(type(k)))
    m, n = matrix.shape[-2:]
    if k is None:
        k = min(6, m, n)

    c = np.mean(matrix, axis=-2)
    norm_matrix = matrix - c

    q = get_approximate_basis(norm_matrix.T, k, niter)
    q_c = q.conjugate()
    b_t = mat_mul(norm_matrix, q_c)
    _, _, v = np.linalg.svd(b_t, full_matrices=False)
    v_c = v.conj().transpose(-2, -1)
    v_c = mat_mul(q, v_c)

    if not norm:
        matrix = mat_mul(matrix, v_c)
    else:
        matrix = mat_mul(norm_matrix, v_c)

    return matrix

def read_adata(file_fold,file_name):
    adata = sc.read_visium(file_fold, count_file=file_name, load_images=True)
    adata.X = adata.X.toarray()
    adata.var_names_make_unique()
    print(adata)
    return adata

def build_her2st_data(path, name):
    cnt_path = os.path.join(path, 'data/ST-cnts', f'{name}.tsv/ut_{name}_stdata_filtered.tsv')
    df_cnt = pd.read_csv(cnt_path, sep='\t', index_col=0)

    pos_path = os.path.join(path, 'data/ST-spotfiles', f'{name}_selection.tsv')
    df_pos = pd.read_csv(pos_path, sep='\t')

    lbl_path = os.path.join(path, 'data/ST-pat/lbl', f'{name}_labeled_coordinates.tsv')
    df_lbl = pd.read_csv(lbl_path, sep='\t')
    df_lbl = df_lbl.dropna(axis=0, how='any')
    df_lbl.loc[df_lbl['label'] == 'undetermined', 'label'] = np.nan
    df_lbl['x'] = (df_lbl['x']+0.5).astype(np.int64)
    df_lbl['y'] = (df_lbl['y']+0.5).astype(np.int64)

    x = df_pos['x'].values
    y = df_pos['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_pos['id'] = ids

    x = df_lbl['x'].values
    y = df_lbl['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_lbl['id'] = ids

    meta_pos = df_cnt.join(df_pos.set_index('id'))
    meta_lbl = df_cnt.join(df_lbl.set_index('id'))

    adata = anndata.AnnData(df_cnt, dtype=np.int64)
    adata.obsm['spatial'] = np.floor(meta_pos[['pixel_x','pixel_y']].values).astype(int)
    adata.obsm['label'] = pd.Categorical(meta_lbl['label']).codes
    print(adata,adata.obsm['label'],adata.obsm['spatial'])
    return adata

def read_label(adata,file_path,lable_name):
    pd_label = pd.read_csv(file_path,sep='\t')
    df_label = pd.DataFrame(pd_label,columns=[lable_name])
    label = pd.Categorical(df_label[lable_name]).codes
    original_labels = pd.categories[label]
    print(label)
    print("\n方法1 - 使用categories映射:")
    print(original_labels)
    adata.obsm['label']=label
    adata.obsm['original_labels']=original_labels
    print(adata)
    return adata


def adata_preprocess_pca(i_adata, min_cells=3, pca_n_comps=300):
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    
    return adata_X


def adata_preprocess_hvg(adata, n_top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    return adata[:, adata.var['highly_variable']].X

def process_adata(adata,pca_dim=1000,k=50):
    #标准化
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000.0)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = (adata.X - adata.X.mean(0)) / (adata.X.std(0) + 1e-15)
    gene_tensor = pca(adata.X, pca_dim)
    adata.obsm["X_pca"] = gene_tensor
    #位置坐标
    position = np.ascontiguousarray(adata.obsm["spatial"]) 
    #计算节点距离
    DIS = squareform(pdist(position))
    adata.obsm["distance"] = DIS
    #k个邻居
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(k)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    #transform adj to symmetrical adj
    adj_k = interaction
    adj_k = adj_k + adj_k.T
    adj_k = np.where(adj_k>1, 1, adj_k)
    # print(adj,adj.shape,np.count_nonzero(adj))
    adata.obsm['adj_k'] = adj_k
    #计算knn节点距离
    DIS_K=np.multiply(DIS,adj_k)
    adata.obsm['distance_k'] = DIS_K
    print(adata)
    return adata

def prepare_data(adata,threshold):
    adj=adata.obsm['distance']
    adj[adj > threshold]=0
    adj_dis = adj.ravel()[np.flatnonzero(adj)]

    adj_dis = 1000-adj_dis
    
    edge_index = sp.coo_matrix(adj)
    values = edge_index.data 
    indices = np.vstack((edge_index.row, edge_index.col)) # 我们真正需要的coo形式 
    edge_index = torch.LongTensor(indices) # PyG框架需要的

    edge_attr = adj_dis     
    edge_attr = torch.tensor(edge_attr).float()
    data = torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr, x=adata.obsm["X_pca"], y = adata.obsm["label"],
                     neighbor_index=edge_index, neighbor_attr=edge_attr)
    return data

def refine(adata,sample_id, pred, shape="hexagon"):
    position = np.ascontiguousarray(adata.obsm["spatial"]) 
    dis = squareform(pdist(position))
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred