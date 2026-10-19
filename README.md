# KNN 鸢尾花分类与可视化分析

课程汇报程序：使用 K 近邻（K-Nearest Neighbors, KNN）对 Iris 鸢尾花数据集进行三分类，
并通过 10 张图表分析模型的选择、决策边界、误差来源和特征标准化的影响。

## 目录结构

```
KNN/
├── knn_iris_analysis.py     # 主程序：数据探索、建模、调参、评估、可视化
├── knn_from_scratch.py      # 仅用 NumPy 手写 KNN，并与 sklearn 对比
├── requirements.txt         # 依赖清单
├── run_all.sh               # 一键运行两个脚本
├── README.md
└── outputs/
    ├── figures/             # 10 张图表（PPT 素材）
    ├── results.json         # 结构化实验结果（便于制作 PPT）
    └── results.txt          # 文本版结果摘要
```

## 环境与运行

项目已在本目录创建 `.venv`（Python 3.14 + scikit-learn 1.9.1）。
若需在新环境重建：

```bash
cd /Users/muyuchen/course/AIoT/KNN
uv venv .venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

运行主程序（生成图 1–8、图 10 及结果文件）：

```bash
.venv/bin/python knn_iris_analysis.py
```

运行手写实现对比（生成图 9）：

```bash
.venv/bin/python knn_from_scratch.py
```

希望生成后弹窗查看图表时，给主程序加 `--show`：

```bash
.venv/bin/python knn_iris_analysis.py --show
```

也可以直接执行 `./run_all.sh` 一次性运行两个脚本。

## 程序流程（可直接对应汇报结构）

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 1. 数据加载与探索 | `load_data()` / `explore_data()` | 加载 Iris，查看类别分布、特征统计，画 pairplot 与相关性热力图 |
| 2. 划分与标准化 | `prepare_data()` | 分层划分 70%/30%，**只在训练集上** `fit` 标准化器 |
| 3. 基线模型 | `baseline_model()` | `KNeighborsClassifier(n_neighbors=5)` |
| 4. k 值选择 | `select_k()` | 5 折分层交叉验证，候选 k=1~30；画 CV 曲线与训练/测试曲线 |
| 5. 最终评估 | `evaluate_final_model()` | 测试集准确率、分类报告、混淆矩阵 |
| 6. 决策边界 | `plot_decision_boundaries()` | 取花瓣长度×花瓣宽度二维子集，对比 k=1/8/80 |
| 7. 标准化分析 | `plot_scaling_effect()` | 计算各特征对距离的贡献占比 |
| 7b. 补充实验 | `wine_scaling_experiment()` | Wine 数据集（13 个特征量纲差异大）标准化前后对比 |
| 8. 邻居投票 | `plot_nearest_neighbors()` | 选最不确定的测试样本，画 k 个最近邻及投票 |
| 9. 手写实现 | `knn_from_scratch.py` | 距离计算→排序→投票，与 sklearn 预测逐条对比 |

## 图表清单

| 文件 | 内容 | 汇报要点 |
|---|---|---|
| `01_pairplot.png` | 四特征两两散点图 + 核密度 | 山鸢尾与另外两类线性可分；花瓣特征区分度最高 |
| `02_correlation_heatmap.png` | 特征相关性热力图 | 花瓣长度与花瓣宽度相关系数 0.96，存在冗余 |
| `03_k_selection_cv.png` | 交叉验证准确率随 k 变化 | CV 选出最佳 k=8，准确率 97.14% |
| `04_train_test_vs_k.png` | 训练/测试准确率随 k 变化 | k 过小训练准确率 100%（过拟合）；k 过大两者都下降（欠拟合） |
| `05_confusion_matrix.png` | 测试集混淆矩阵 | 4 个维吉尼亚鸢尾被误判为变色鸢尾 |
| `06_decision_boundary.png` | k=1 / 8 / 80 决策边界 | k=1 边界破碎；k=80 边界几乎抹平所有局部结构 |
| `07_scaling_effect.png` | 各特征对距离的贡献占比 | 原始特征下花瓣长度占 67.1%；标准化后四特征各占 25% |
| `08_nearest_neighbors_demo.png` | 一个测试样本的 8 个最近邻 | 邻居投票 5:3，模型把维吉尼亚鸢尾误判为变色鸢尾 |
| `09_scratch_vs_sklearn.png` | 手写 KNN 与 sklearn 对比 | 多数投票实现与 sklearn 预测一致率 100% |
| `10_wine_scaling.png` | Wine 数据集标准化前后 | CV 准确率从 71.8% 提升到 96.8%，量纲差异大时标准化至关重要 |

## 核心结果

- **最佳 k = 8**，5 折交叉验证准确率 **97.14%**，测试集准确率 **91.11%**（41/45）。
- 测试集的 4 个错误全部是「维吉尼亚鸢尾 → 变色鸢尾」，说明两类在花瓣特征上存在重叠。
- 最近邻演示样本的投票为「变色鸢尾 5 票 vs 维吉尼亚鸢尾 3 票」，直观展示了误判原因。
- Iris 各特征量纲接近，标准化前后 CV 准确率相同；但在量纲差异大的 Wine 数据集上，
  标准化后 CV 准确率从 **71.8%** 提升到 **96.8%**。

## 汇报时值得强调的点

1. KNN 是**懒惰学习**：`fit()` 只保存训练数据，预测时才计算距离。
2. 三要素：**距离度量、k 值、决策规则**（多数投票 / 距离加权）。
3. **标准化必须做**，且只能用训练集拟合 `StandardScaler`，避免测试集信息泄漏。
4. **k 的选择不能看测试集**：本程序用交叉验证选 k，测试集只用于最终评估。
5. 测试集只有 45 个样本，1 个样本对应约 2.2 个百分点，因此测试曲线会有波动；
   做模型选择时应更信任交叉验证结果。
6. KNN 的局限：预测复杂度 O(n·d)、内存占用大、高维下距离失效（维度灾难）、
   对类别不平衡敏感。

## 可扩展方向

- `weights="distance"` 距离加权投票（手写脚本中已实现，测试准确率 93.33%）。
- 尝试曼哈顿距离、余弦距离等不同度量。
- 用 PCA 把 4 维特征降到 2 维，观察全特征模型的决策边界。
- 在 `GridSearchCV` 中联合搜索 `n_neighbors`、`weights`、`p`。
