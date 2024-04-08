# Hierarchical Position Embedding of Graphs with Landmarks and Clustering for Link Prediction
"Hierarchical Position Embedding of Graphs with Landmarks and Clustering for Link Prediction" <br> The International World Wide Web Conference (WWW) 2024, Accepted. <br>


## Environment
```bash
pytorch 1.13.1+cu117
```

### Requirements
```bash
pip install -r requirements.txt
```

### OGB - CITATION2, COLLAB, DDI
```bash
 cd ogb
 python clustering_{data}.py k (eta in the paper)
 python {data}/main.py
```


### Others - Pubmed, Cora, Citeseer, Facebook
##### Please download data from this: [download](https://drive.google.com/drive/folders/1IJGklD1nvsAOfDUQDJF0by7YxtGrprgS?usp=sharing)
```bash
 cd others
 python clustering.py cora or pubmed or citeseer k (eta in the paper)
 python clustering_fb.py facebook k (eta in the paper)
 python main.py --dataset {dataset}
```

## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite our [paper](https://arxiv.org/pdf/2402.08174.pdf).
```
@InProceedings{
  title = 	 Hierarchical Position Embedding of Graphs with Landmarks and Clustering for Link Prediction},
  author =       {Minsang Kim, Seungjun Baek},
  booktitle = 	 {The ACM Web Conference (WWW)},
  year = 	 {2024},
  series = 	 {Proceedings of The ACM Web Conference},
  month = 	 {13--17 May},
  publisher =    {The International World Wide Web Conference (WWW)},
  pdf = 	 {https://arxiv.org/pdf/2402.08174.pdf},
  abstract = 	 {Learning positional information of nodes in a graph is important for link prediction tasks. We propose a representation of positional information using representative nodes called landmarks. A small number of nodes with high degree centrality are selected as landmarks, which serve as reference points for the nodes' positions. We justify this selection strategy for well-known random graph models, and derive closed-form bounds on the average path lengths involving landmarks. In a model for scale-free networks, we prove that landmarks provide asymptotically exact information on inter-node distances. We apply theoretical insights to practical networks, and propose Hierarchical Position embedding with Landmarks and Clustering (HPLC). HPLC combines landmark selection and graph clustering, where the graph is partitioned into densely connected clusters in which nodes with the highest degree are selected as landmarks. HPLC leverages the positional information of nodes based on landmarks at various levels of hierarchy such as nodes' distances to landmarks, inter-landmark distances and hierarchical grouping of clusters. Experiments show that HPLC achieves state-of-the-art performances of link prediction on various datasets in terms of HIT@K, MRR, and AUC.}
  }
```
