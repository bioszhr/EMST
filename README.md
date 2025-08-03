# EMST: an interpretable multi-modal model for spatial transcriptomic data analysis
## Overview
Here we propose EMST, an interpretable multi-modal spatial transcriptomic data augmentation model that uses histopathological images to learn representations for gene expression data. We first extract morphological features of spots using a pre-trained large biological model, and then integrate morphological and gene features through a co-attention-based multi-modal fusion module. The fused features undergo an interpretable contrastive learning model to complete feature learning and enhance original gene expression data.<br>
<div align="center">
  <img src="https://github.com/bioszhr/EMST/main/results/figures/Figure 1.pdf">
</div>
## Requirements
Please ensure that all the libraries below are successfully installed:<br>
-Python 3.7<br>
-torch 1.12.0<br>
-torch-geometric 1.7.0<br>
-faiss 1.7.0<br>
## Run EMST on the example data.
If you wanna run **the breast invasive carcinoma (BRCA)** dataset,you should change the relevant path to your own file path, and run ***EMST.ipynb***.<br>
### Output
The output results will be stored in the dir ***results***.
### Datasets
The spatial transcriptomics datasets utilized in this study can be downloaded from: (1) The breast invasive carcinoma
 (BRCA) dataset https://www.10xgenomics.com/resourcs/datasets. (2) human HER2-positive breast
 tumor ST data https://github.com/almaan/HER2st/.
