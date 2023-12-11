## Environment
```bash
pytorch 1.10.0
```

### Requirements
```bash
pip install -r requirements.txt
```

### OGB - CITATION2, COLLAB, DDI
```bash
 python clustering_{data}.py k (eta in the paper)
 python {data}/main.py
```


### Others - Pubmed, Cora, Citeseer, Facebook
```bash
 python clustering.py cora or pubmed or citeseer k (eta in the paper)
 python clustering_fb.py facebook k (eta in the paper)
 python main.py --dataset {dataset}
```
