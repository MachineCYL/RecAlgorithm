## 协同过滤算法（Collaborative Filtering）

基于pyspark的ALS模型（交替最小二乘矩阵分解）来实现

#### 运行指令

- 进行ALS模型训练和预测

```
python train_cf.py  --data_path "./data/ratings.csv" --train_flag True
```

- 不训练，直接进行ALS模型预测
```
python train_cf.py  --data_path "./data/ratings.csv"
```