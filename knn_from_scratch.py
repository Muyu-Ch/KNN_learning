#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从零实现 KNN（仅用 NumPy），并与 scikit-learn 的结果对比
================================================================

目的：配合汇报讲解 KNN 的内部计算过程——
      距离计算 → 排序取前 k → 多数投票（或距离加权投票）。

运行方式：
    .venv/bin/python knn_from_scratch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3
K = 5
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CLASS_CN = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, AXIS, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "grid.color": GRID,
        "legend.frameon": False,
    }
)


class ScratchKNN:
    """K 近邻分类器的最小实现。

    参数
    ----
    n_neighbors : int
        邻居个数 k。
    weights : {"uniform", "distance"}
        uniform   -> 多数投票；
        distance  -> 距离加权投票，权重 = 1 / 距离。
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        # KNN 是“懒惰学习”：训练阶段只保存数据，不做任何参数学习
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _predict_one(self, x):
        # 1. 计算待测样本到所有训练样本的欧氏距离
        distances = np.sqrt(((self.X_train_ - x) ** 2).sum(axis=1))

        # 2. 排序并取最近的 k 个（argpartition 先粗筛，再精确排序）
        k = min(self.n_neighbors, len(self.X_train_))
        candidates = np.argpartition(distances, k - 1)[:k]
        neighbors = candidates[np.argsort(distances[candidates])]

        # 3. 统计投票
        labels = self.y_train_[neighbors]
        if self.weights == "uniform":
            scores = np.array(
                [np.sum(labels == c) for c in self.classes_], dtype=float
            )
        else:  # 距离加权：越近的邻居票越重
            w = 1.0 / (distances[neighbors] + 1e-12)
            scores = np.array(
                [w[labels == c].sum() for c in self.classes_], dtype=float
            )

        return self.classes_[np.argmax(scores)], distances[neighbors], labels

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x)[0] for x in X])


def main():
    print("=" * 72)
    print("从零实现 KNN 并与 scikit-learn 对比")
    print("=" * 72)

    # ---- 数据准备（与主程序保持一致）----
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- 三个模型 ----
    sklearn_model = KNeighborsClassifier(n_neighbors=K).fit(X_train_s, y_train)
    scratch_uniform = ScratchKNN(n_neighbors=K, weights="uniform").fit(
        X_train_s, y_train
    )
    scratch_distance = ScratchKNN(n_neighbors=K, weights="distance").fit(
        X_train_s, y_train
    )

    pred_sklearn = sklearn_model.predict(X_test_s)
    pred_uniform = scratch_uniform.predict(X_test_s)
    pred_distance = scratch_distance.predict(X_test_s)

    acc_sklearn = accuracy_score(y_test, pred_sklearn)
    acc_uniform = accuracy_score(y_test, pred_uniform)
    acc_distance = accuracy_score(y_test, pred_distance)
    agreement = np.mean(pred_sklearn == pred_uniform)

    print(f"\nk = {K}，测试集 {len(y_test)} 个样本")
    print(f"scikit-learn KNeighborsClassifier 准确率: {acc_sklearn:.4f}")
    print(f"手写 KNN（多数投票）准确率:              {acc_uniform:.4f}")
    print(f"手写 KNN（距离加权）准确率:              {acc_distance:.4f}")
    print(f"手写实现与 sklearn 预测一致率:           {agreement:.2%}")

    # ---- 展示一个测试样本的完整计算过程 ----
    demo_idx = 0
    x_demo = X_test_s[demo_idx]
    pred, distances, labels = scratch_uniform._predict_one(x_demo)

    print("\n" + "-" * 72)
    print("计算过程演示（测试集第 1 个样本）")
    print("-" * 72)
    print(f"真实类别：{CLASS_CN[int(y_test[demo_idx])]}")
    print(f"到 {len(X_train_s)} 个训练样本的距离已计算，最近的 {K} 个：")
    for rank, (d, label) in enumerate(zip(distances, labels), start=1):
        print(f"  第 {rank} 近：距离 {d:.4f} → {CLASS_CN[int(label)]}")
    votes = np.bincount(labels, minlength=3)
    print(
        "投票结果："
        + "，".join(f"{name} {int(n)} 票" for name, n in zip(CLASS_CN, votes))
    )
    print(f"预测类别：{CLASS_CN[int(pred)]}")

    # ---- 图：三种实现准确率对比 ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    names = ["scikit-learn", "手写实现\n（多数投票）", "手写实现\n（距离加权）"]
    accs = [acc_sklearn, acc_uniform, acc_distance]
    colors = [BLUE, ORANGE, AQUA]

    bars = ax.bar(
        range(3), accs, width=0.52, color=colors,
        edgecolor="#fcfcfb", linewidth=2, zorder=2,
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.015,
            f"{acc:.2%}", ha="center", va="bottom",
            fontsize=13, fontweight="semibold", color=INK,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("测试集准确率")
    ax.set_title(f"图 9  手写 KNN 与 scikit-learn 对比（k = {K}）")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=10)

    path = FIG_DIR / "09_scratch_vs_sklearn.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"\n[图] {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
