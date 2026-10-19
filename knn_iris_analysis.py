#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 K 近邻（K-Nearest Neighbors, KNN）的鸢尾花分类与可视化分析
===================================================================

案例：Iris 鸢尾花数据集（150 个样本、4 个特征、3 个类别）
目标：用 KNN 完成三分类，并通过可视化分析模型行为：
      1. 数据探索：特征分布、类别可分性、相关性
      2. k 值选择：交叉验证准确率曲线 + 训练/测试准确率曲线
      3. 决策边界：不同 k 值下模型如何划分特征空间
      4. 混淆矩阵：模型错在哪里
      5. 特征标准化：距离型模型为何必须标准化
      6. 最近邻演示：一个测试样本的 k 个邻居如何投票

运行方式：
    .venv/bin/python knn_iris_analysis.py            # 生成全部图表到 outputs/figures
    .venv/bin/python knn_iris_analysis.py --show     # 生成后弹窗显示
"""

import argparse

# 先解析参数，再决定 matplotlib 后端（无 --show 时用无界面后端批量保存图片）
parser = argparse.ArgumentParser(description="KNN 鸢尾花分类与可视化分析")
parser.add_argument("--show", action="store_true", help="生成图表后弹窗显示")
ARGS = parser.parse_args()

import json
from pathlib import Path

import matplotlib

if not ARGS.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.3
K_RANGE = list(range(1, 31))          # 候选 k 值：1 ~ 30
CV_FOLDS = 5                          # 交叉验证折数
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体与绘图风格
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
        "axes.labelsize": 11,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)

# 配色：前三个类别色为经过色觉障碍校验的分类色
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"

CLASS_COLORS = [BLUE, ORANGE, AQUA]
# 决策区域使用的浅色底（与类别色同色相，降低饱和度）
CLASS_REGION_COLORS = ["#dbe9fb", "#fbe3d8", "#d3f1e6"]

FEATURE_CN = [
    "萼片长度 (cm)",
    "萼片宽度 (cm)",
    "花瓣长度 (cm)",
    "花瓣宽度 (cm)",
]
CLASS_CN = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]

BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#184f95"]
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def style_axes(ax, grid_axis="both"):
    """统一坐标轴样式：去掉上/右框线，保留浅色网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=10)


def save_fig(fig, filename):
    """保存图片并打印相对路径。"""
    path = FIG_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  [图] {path.relative_to(BASE_DIR)}")
    return path


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# 1. 数据加载与探索
# ---------------------------------------------------------------------------
def load_data():
    """加载 Iris 数据集，返回带中文列名的 DataFrame 与特征/标签。"""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns=dict(zip(iris.feature_names, FEATURE_CN)))
    df = df.rename(columns={"target": "类别编码"})
    df["品种"] = pd.Categorical(
        [CLASS_CN[i] for i in df["类别编码"]], categories=CLASS_CN, ordered=True
    )

    X = df[FEATURE_CN].copy()
    y = df["类别编码"].copy()
    return df, X, y, iris


def explore_data(df, X, y, iris):
    """输出数据概况并绘制 pairplot、相关性热力图。"""
    section("1. 数据加载与探索")

    print(f"样本数：{len(df)}，特征数：{X.shape[1]}，类别数：{y.nunique()}")
    print("\n各类别样本数：")
    counts = df["品种"].value_counts().reindex(CLASS_CN)
    for name, n in counts.items():
        print(f"  {name}: {n} 个")
    print("\n特征统计：")
    print(X.describe().round(2).to_string())
    print(f"\n缺失值总数：{int(X.isna().sum().sum())}")

    # ---- 图 1：特征两两散点图（按品种着色）----
    g = sns.pairplot(
        df[FEATURE_CN + ["品种"]],
        hue="品种",
        hue_order=CLASS_CN,
        palette=CLASS_COLORS,
        diag_kind="kde",
        plot_kws={
            "s": 36,
            "alpha": 0.85,
            "edgecolor": "white",
            "linewidth": 0.6,
        },
        diag_kws={"linewidth": 2, "fill": False},
        height=2.15,
    )
    g.figure.suptitle("图 1  Iris 特征两两分布：山鸢尾与另外两类线性可分", y=1.02)
    if g._legend is not None:
        g._legend.set_title("品种")
    save_fig(g.figure, "01_pairplot.png")

    # ---- 图 2：特征相关性热力图 ----
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    corr = X.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=BLUE_SEQ,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.5,
        linecolor="#fcfcfb",
        cbar_kws={"shrink": 0.8, "label": "相关系数"},
        ax=ax,
    )
    ax.set_title("图 2  特征相关性：花瓣长度与花瓣宽度高度相关")
    ax.tick_params(colors=INK_2, labelsize=10)
    save_fig(fig, "02_correlation_heatmap.png")

    return counts


# ---------------------------------------------------------------------------
# 2. 划分数据集与标准化
# ---------------------------------------------------------------------------
def prepare_data(X, y):
    """分层划分训练/测试集，并只在训练集上拟合标准化器。"""
    section("2. 数据划分与特征标准化")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"训练集：{len(X_train)} 个样本（{(1 - TEST_SIZE) * 100:.0f}%）")
    print(f"测试集：{len(X_test)} 个样本（{TEST_SIZE * 100:.0f}%）")
    print("分层抽样保证训练/测试集中三类比例一致。")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n标准化后训练集各特征均值（应接近 0）：")
    print(np.round(X_train_scaled.mean(axis=0), 4))
    print("标准化后训练集各特征标准差（应接近 1）：")
    print(np.round(X_train_scaled.std(axis=0), 4))

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 3. 基线模型
# ---------------------------------------------------------------------------
def baseline_model(X_train_scaled, X_test_scaled, y_train, y_test, k=5):
    """用默认参数（k=5）建立基线模型。"""
    section(f"3. 基线模型（k = {k}）")

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率：{acc:.4f}（{int(acc * len(y_test))}/{len(y_test)}）")
    print("\n分类报告：")
    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_CN, digits=4
        )
    )
    return model, y_pred, acc


# ---------------------------------------------------------------------------
# 4. k 值选择
# ---------------------------------------------------------------------------
def select_k(X_train_scaled, X_test_scaled, y_train, y_test):
    """交叉验证选择 k，并绘制 CV 曲线与训练/测试曲线。"""
    section("4. k 值选择（5 折分层交叉验证）")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    cv_means, cv_stds = [], []
    for k in K_RANGE:
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k),
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())

    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)
    best_k = K_RANGE[int(np.argmax(cv_means))]
    print(f"候选 k：{K_RANGE[0]} ~ {K_RANGE[-1]}")
    print(f"最佳 k = {best_k}，交叉验证准确率 = {cv_means.max():.4f}")

    # ---- 图 3：交叉验证准确率随 k 变化 ----
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(
        K_RANGE,
        cv_means,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="5 折交叉验证准确率",
    )
    ax.fill_between(
        K_RANGE,
        cv_means - cv_stds,
        cv_means + cv_stds,
        color=BLUE,
        alpha=0.14,
        linewidth=0,
        label="±1 个标准差",
    )
    ax.scatter(
        [best_k],
        [cv_means.max()],
        s=110,
        color=ORANGE,
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        f"最佳 k = {best_k}\n准确率 {cv_means.max():.3f}",
        xy=(best_k, cv_means.max()),
        xytext=(best_k + 2.5, cv_means.max() - 0.035),
        fontsize=10,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
    )
    ax.set_xlabel("k 值（邻居个数）")
    ax.set_ylabel("交叉验证准确率")
    ax.set_ylim(min(cv_means) - 0.03, 1.005)
    ax.set_xticks(range(1, 31, 2))
    ax.legend(loc="lower right")
    ax.set_title("图 3  k 值选择：交叉验证准确率随 k 的变化")
    style_axes(ax)
    save_fig(fig, "03_k_selection_cv.png")

    # ---- 图 4：训练集/测试集准确率随 k 变化（过拟合/欠拟合）----
    train_acc, test_acc = [], []
    for k in K_RANGE:
        model = KNeighborsClassifier(n_neighbors=k).fit(
            X_train_scaled, y_train
        )
        train_acc.append(model.score(X_train_scaled, y_train))
        test_acc.append(model.score(X_test_scaled, y_test))

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(
        K_RANGE, train_acc, color=BLUE, linewidth=2,
        marker="o", markersize=4, markeredgecolor="white",
        markeredgewidth=0.8, label="训练集准确率",
    )
    ax.plot(
        K_RANGE, test_acc, color=ORANGE, linewidth=2,
        marker="o", markersize=4, markeredgecolor="white",
        markeredgewidth=0.8, label="测试集准确率",
    )
    ax.axvline(best_k, color=MUTED, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(
        best_k + 0.5, 0.885, f"最佳 k = {best_k}",
        color=INK_2, fontsize=10,
    )
    ax.set_xlabel("k 值（邻居个数）")
    ax.set_ylabel("准确率")
    ax.set_xticks(range(1, 31, 2))
    ax.set_ylim(0.84, 1.01)
    ax.legend(loc="lower right")
    ax.text(
        0.02, 0.025,
        "测试集仅用于最终评估，不参与选 k",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9, color=MUTED,
    )
    ax.set_title(
        "图 4  训练/测试准确率随 k 变化（测试集仅 45 个样本，曲线有波动）"
    )
    style_axes(ax)
    save_fig(fig, "04_train_test_vs_k.png")

    return best_k, cv_means.max(), cv_means, cv_stds


# ---------------------------------------------------------------------------
# 5. 最终模型评估
# ---------------------------------------------------------------------------
def evaluate_final_model(
    X_train_scaled, X_test_scaled, y_train, y_test, best_k
):
    """训练最终模型，输出指标并绘制混淆矩阵。"""
    section(f"5. 最终模型评估（k = {best_k}）")

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_CN, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"测试集准确率：{test_acc:.4f}（{int(test_acc * len(y_test))}/{len(y_test)}）")
    print("\n分类报告：")
    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_CN, digits=4
        )
    )
    print("混淆矩阵（行=真实类别，列=预测类别）：")
    print(pd.DataFrame(cm, index=CLASS_CN, columns=CLASS_CN).to_string())

    # ---- 图 5：混淆矩阵 ----
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=BLUE_SEQ,
        cbar=False,
        square=True,
        linewidths=2,
        linecolor="#fcfcfb",
        xticklabels=CLASS_CN,
        yticklabels=CLASS_CN,
        annot_kws={"fontsize": 16, "fontweight": "semibold"},
        ax=ax,
    )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(
        f"图 5  测试集混淆矩阵（k = {best_k}，准确率 {test_acc:.1%}）"
    )
    ax.tick_params(colors=INK_2, labelsize=10)
    save_fig(fig, "05_confusion_matrix.png")

    return model, y_pred, test_acc, report, cm


# ---------------------------------------------------------------------------
# 6. 决策边界可视化
# ---------------------------------------------------------------------------
def plot_decision_boundaries(X_train, X_test, y_train, y_test, best_k):
    """在二维特征子集（花瓣长度、花瓣宽度）上绘制不同 k 的决策边界。

    注意：这里为了可视化，只用两个花瓣特征重新训练模型；
    最终模型仍使用全部 4 个特征。
    """
    section("6. 决策边界可视化（二维特征子集：花瓣长度 × 花瓣宽度）")

    feature_idx = [2, 3]  # 花瓣长度、花瓣宽度
    X2_train = X_train.iloc[:, feature_idx].to_numpy()
    X2_test = X_test.iloc[:, feature_idx].to_numpy()
    feature_short = ["花瓣长度 (cm)", "花瓣宽度 (cm)"]

    ks = [1, best_k, 80]
    fig, axes = plt.subplots(
        1, len(ks), figsize=(15.5, 4.8), constrained_layout=True
    )

    for ax, k in zip(axes, ks):
        scaler = StandardScaler().fit(X2_train)
        Xtr = scaler.transform(X2_train)
        Xte = scaler.transform(X2_test)

        model = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y_train)
        train_score = model.score(Xtr, y_train)
        test_score = model.score(Xte, y_test)

        # 在原始特征尺度上生成网格，再标准化后送入模型预测
        x_min, x_max = X2_train[:, 0].min() - 0.4, X2_train[:, 0].max() + 0.4
        y_min, y_max = X2_train[:, 1].min() - 0.3, X2_train[:, 1].max() + 0.3
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 350),
            np.linspace(y_min, y_max, 350),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(scaler.transform(grid)).reshape(xx.shape)

        ax.contourf(
            xx, yy, Z,
            levels=np.arange(-0.5, len(CLASS_CN) + 0.5, 1),
            colors=CLASS_REGION_COLORS,
            alpha=0.9,
            zorder=0,
        )
        # 训练样本点
        for ci, (cname, color) in enumerate(zip(CLASS_CN, CLASS_COLORS)):
            mask = y_train == ci
            ax.scatter(
                X2_train[mask, 0], X2_train[mask, 1],
                s=46, color=color, edgecolor="white", linewidth=1.0,
                label=cname if k == ks[0] else None,
                zorder=3,
            )
        # 测试样本点：空心标记，便于观察边界处的错分
        ax.scatter(
            X2_test[:, 0], X2_test[:, 1],
            s=52, facecolor="none", edgecolor=INK, linewidth=1.2,
            marker="s", label="测试样本" if k == ks[0] else None,
            zorder=4,
        )

        ax.set_title(
            f"k = {k}\n训练 {train_score:.1%} · 测试 {test_score:.1%}",
            fontsize=12,
        )
        ax.set_xlabel(feature_short[0])
        if k == ks[0]:
            ax.set_ylabel(feature_short[1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        style_axes(ax, grid_axis="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10),
        fontsize=10,
    )
    fig.suptitle(
        "图 6  不同 k 值下的决策边界：k=1 边界破碎（过拟合），k=80 局部结构被抹平（欠拟合）",
        fontsize=13, fontweight="semibold",
    )
    save_fig(fig, "06_decision_boundary.png")


# ---------------------------------------------------------------------------
# 7. 特征标准化的影响
# ---------------------------------------------------------------------------
def _distance_contributions(X):
    """计算每个特征对样本间欧氏距离平方的贡献占比（%）。"""
    diff = X[:, None, :] - X[None, :, :]
    contrib = (diff ** 2).sum(axis=(0, 1))
    return contrib / contrib.sum() * 100


def plot_scaling_effect(X_train, y_train, X_train_scaled, best_k):
    """说明标准化的作用：观察各特征对距离的贡献是否均衡。"""
    section("7. 特征标准化对 KNN 的影响")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    raw_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=best_k),
        X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    scaled_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=best_k),
        X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    print(f"未标准化：CV 准确率 {raw_scores.mean():.4f} ± {raw_scores.std():.4f}")
    print(f"标准化后：CV 准确率 {scaled_scores.mean():.4f} ± {scaled_scores.std():.4f}")

    raw_contrib = _distance_contributions(X_train.to_numpy())
    scaled_contrib = _distance_contributions(X_train_scaled)
    print("\n各特征对距离的贡献占比（原始特征）：")
    for name, v in zip(FEATURE_CN, raw_contrib):
        print(f"  {name}: {v:.1f}%")
    print("各特征对距离的贡献占比（标准化后）：")
    for name, v in zip(FEATURE_CN, scaled_contrib):
        print(f"  {name}: {v:.1f}%")
    print("说明：KNN 依赖距离，量纲大的特征会主导距离计算，标准化可消除这种偏差。")

    # ---- 图 7：标准化前后各特征的距离贡献 ----
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    y_pos = np.arange(len(FEATURE_CN))[::-1]
    bar_h = 0.36

    ax.barh(
        y_pos + bar_h / 2, raw_contrib, height=bar_h,
        color=ORANGE, edgecolor="#fcfcfb", linewidth=1.5,
        label="未标准化（原始特征）", zorder=2,
    )
    ax.barh(
        y_pos - bar_h / 2, scaled_contrib, height=bar_h,
        color=BLUE, edgecolor="#fcfcfb", linewidth=1.5,
        label="标准化后（StandardScaler）", zorder=2,
    )
    for y, v in zip(y_pos + bar_h / 2, raw_contrib):
        ax.text(v + 1.2, y, f"{v:.1f}%", va="center", fontsize=10, color=INK_2)
    for y, v in zip(y_pos - bar_h / 2, scaled_contrib):
        ax.text(v + 1.2, y, f"{v:.1f}%", va="center", fontsize=10, color=INK_2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_CN, fontsize=10.5)
    ax.set_xlim(0, 82)
    ax.set_xlabel("该特征对样本间欧氏距离平方的贡献占比（%）")
    ax.set_title("图 7  标准化前后各特征对距离的贡献：从“一家独大”到“雨露均沾”")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.text(
        0.985, 0.03,
        f"本例 CV 准确率：原始 {raw_scores.mean():.1%} / 标准化 {scaled_scores.mean():.1%}\n"
        "Iris 各特征量纲接近，准确率差异小；量纲差异大的案例见图 10",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=MUTED, linespacing=1.5,
    )
    style_axes(ax, grid_axis="x")
    save_fig(fig, "07_scaling_effect.png")

    return raw_scores.mean(), scaled_scores.mean()


# ---------------------------------------------------------------------------
# 7b. 补充实验：量纲差异大的 Wine 数据集
# ---------------------------------------------------------------------------
def wine_scaling_experiment():
    """在 Wine 数据集上对比标准化前后的 KNN，突出标准化的必要性。

    Wine 的 13 个特征量纲差异悬殊（proline 跨度约 1402，hue 仅约 1.23），
    未标准化时距离会被大尺度特征完全主导。
    """
    section("7b. 补充实验：Wine 数据集上的标准化对比")

    wine = load_wine(as_frame=True)
    Xw, yw = wine.data, wine.target
    ranges = (Xw.max() - Xw.min()).sort_values(ascending=False)
    print(f"样本数：{len(Xw)}，特征数：{Xw.shape[1]}，类别数：{yw.nunique()}")
    print("特征取值范围差异（前 5）：")
    for name, r in ranges.head(5).items():
        print(f"  {name}: {r:.2f}")

    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        Xw, yw, test_size=TEST_SIZE, stratify=yw, random_state=RANDOM_STATE
    )
    scaler_w = StandardScaler().fit(Xw_train)
    Xw_train_s = scaler_w.transform(Xw_train)
    Xw_test_s = scaler_w.transform(Xw_test)

    k_wine = 5
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    raw_cv = cross_val_score(
        KNeighborsClassifier(n_neighbors=k_wine),
        Xw_train, yw_train, cv=cv, scoring="accuracy", n_jobs=-1,
    ).mean()
    scaled_cv = cross_val_score(
        KNeighborsClassifier(n_neighbors=k_wine),
        Xw_train_s, yw_train, cv=cv, scoring="accuracy", n_jobs=-1,
    ).mean()
    raw_test = KNeighborsClassifier(n_neighbors=k_wine).fit(
        Xw_train, yw_train
    ).score(Xw_test, yw_test)
    scaled_test = KNeighborsClassifier(n_neighbors=k_wine).fit(
        Xw_train_s, yw_train
    ).score(Xw_test_s, yw_test)

    print(f"\nk = {k_wine}")
    print(f"未标准化：CV {raw_cv:.4f}，测试集 {raw_test:.4f}")
    print(f"标准化后：CV {scaled_cv:.4f}，测试集 {scaled_test:.4f}")

    # ---- 图 10：Wine 数据集标准化前后对比 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    groups = ["5 折交叉验证", "测试集"]
    x = np.arange(len(groups))
    width = 0.34

    bars_raw = ax.bar(
        x - width / 2, [raw_cv, raw_test], width,
        color=ORANGE, edgecolor="#fcfcfb", linewidth=2,
        label="未标准化", zorder=2,
    )
    bars_scaled = ax.bar(
        x + width / 2, [scaled_cv, scaled_test], width,
        color=BLUE, edgecolor="#fcfcfb", linewidth=2,
        label="标准化后", zorder=2,
    )
    for bar, v in zip(bars_raw, [raw_cv, raw_test]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.015,
            f"{v:.1%}", ha="center", va="bottom",
            fontsize=12.5, fontweight="semibold", color=INK,
        )
    for bar, v in zip(bars_scaled, [scaled_cv, scaled_test]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.015,
            f"{v:.1%}", ha="center", va="bottom",
            fontsize=12.5, fontweight="semibold", color=INK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("准确率")
    ax.set_title(
        f"图 10  Wine 数据集：标准化让 KNN 准确率大幅提升（k = {k_wine}）",
        pad=26,
    )
    ax.text(
        0.5, 1.02,
        "Wine 特征量纲差异悬殊："
        f"{ranges.index[0]} 跨度 {ranges.iloc[0]:.0f}，"
        f"{ranges.index[-1]} 仅 {ranges.iloc[-1]:.2f}",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color=MUTED,
    )
    ax.legend(loc="upper left", fontsize=9.5)
    style_axes(ax, grid_axis="y")
    save_fig(fig, "10_wine_scaling.png")

    return {
        "k": k_wine,
        "n_samples": int(len(Xw)),
        "n_features": int(Xw.shape[1]),
        "raw_cv_accuracy": float(raw_cv),
        "scaled_cv_accuracy": float(scaled_cv),
        "raw_test_accuracy": float(raw_test),
        "scaled_test_accuracy": float(scaled_test),
        "largest_feature_range": {
            "name": ranges.index[0], "range": float(ranges.iloc[0])
        },
        "smallest_feature_range": {
            "name": ranges.index[-1], "range": float(ranges.iloc[-1])
        },
    }


# ---------------------------------------------------------------------------
# 8. 最近邻投票过程可视化
# ---------------------------------------------------------------------------
def plot_nearest_neighbors(X_train, X_test, y_train, y_test, best_k):
    """选择一个最“犹豫”的测试样本，画出它的 k 个最近邻及投票结果。"""
    section(f"8. 最近邻投票过程可视化（选择最不确定的测试样本，k = {best_k}）")

    feature_idx = [2, 3]
    X2_train = X_train.iloc[:, feature_idx].to_numpy()
    X2_test = X_test.iloc[:, feature_idx].to_numpy()
    feature_short = ["花瓣长度 (cm)", "花瓣宽度 (cm)"]

    scaler = StandardScaler().fit(X2_train)
    Xtr = scaler.transform(X2_train)
    Xte = scaler.transform(X2_test)

    model = KNeighborsClassifier(n_neighbors=best_k).fit(Xtr, y_train)
    proba = model.predict_proba(Xte)
    query_idx = int(np.argmin(proba.max(axis=1)))  # 预测置信度最低的样本
    query_scaled = Xte[[query_idx]]
    query_raw = X2_test[query_idx]

    distances, neighbor_idx = model.kneighbors(query_scaled)
    distances = distances[0]
    neighbor_idx = neighbor_idx[0]
    neighbor_labels = y_train.to_numpy()[neighbor_idx]
    neighbor_points = X2_train[neighbor_idx]

    pred_label = int(model.predict(query_scaled)[0])
    true_label = int(y_test.to_numpy()[query_idx])
    votes = np.bincount(neighbor_labels, minlength=len(CLASS_CN))

    print(f"查询样本索引：{query_idx}")
    print(f"真实类别：{CLASS_CN[true_label]}；预测类别：{CLASS_CN[pred_label]}")
    print("最近邻距离（标准化空间）：")
    for rank, (d, label) in enumerate(
        zip(distances, neighbor_labels), start=1
    ):
        print(f"  第 {rank} 近：距离 {d:.3f} → {CLASS_CN[label]}")
    print(
        "投票统计：" +
        "；".join(
            f"{name} {int(n)} 票" for name, n in zip(CLASS_CN, votes)
        )
    )

    # ---- 图 8：最近邻与投票 ----
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)

    # 全部训练样本作为背景
    for ci, (cname, color) in enumerate(zip(CLASS_CN, CLASS_COLORS)):
        mask = y_train.to_numpy() == ci
        ax.scatter(
            X2_train[mask, 0], X2_train[mask, 1],
            s=22, color=color, alpha=0.22,
            edgecolor="none", zorder=1,
        )
        ax.scatter(
            [], [], s=60, color=color, label=cname,
            edgecolor="white", linewidth=0.8,
        )

    # 从查询样本到 k 个邻居的连线
    for nx, ny in neighbor_points:
        ax.plot(
            [query_raw[0], nx], [query_raw[1], ny],
            color=MUTED, linewidth=0.9, alpha=0.65,
            zorder=2,
        )

    # 高亮 k 个邻居
    for ci, color in enumerate(CLASS_COLORS):
        mask = neighbor_labels == ci
        if mask.any():
            ax.scatter(
                neighbor_points[mask, 0], neighbor_points[mask, 1],
                s=120, color=color, edgecolor="white", linewidth=1.8,
                zorder=4,
            )

    # 查询样本用星形标记
    ax.scatter(
        query_raw[0], query_raw[1],
        s=360, marker="*", color=INK,
        edgecolor="white", linewidth=1.2, zorder=5,
    )
    ax.annotate(
        "查询样本", xy=(query_raw[0], query_raw[1]),
        xytext=(query_raw[0] + 0.55, query_raw[1] + 0.28),
        fontsize=10, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.1),
        zorder=6,
    )

    vote_text = (
        f"真实类别：{CLASS_CN[true_label]}\n"
        f"预测类别：{CLASS_CN[pred_label]}\n"
        + "  ".join(f"{name} {int(n)}票" for name, n in zip(CLASS_CN, votes))
    )
    ax.text(
        0.03, 0.97, vote_text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10.5, color=INK, linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fcfcfb", edgecolor=AXIS, linewidth=1.0,
        ),
        zorder=7,
    )

    ax.set_xlabel(feature_short[0])
    ax.set_ylabel(feature_short[1])
    ax.set_title(
        f"图 8  最不确定测试样本的 {best_k} 个最近邻与投票过程",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9.5)
    style_axes(ax)
    save_fig(fig, "08_nearest_neighbors_demo.png")

    return {
        "query_index": query_idx,
        "true_label": CLASS_CN[true_label],
        "pred_label": CLASS_CN[pred_label],
        "votes": {name: int(n) for name, n in zip(CLASS_CN, votes)},
        "distances": [round(float(d), 4) for d in distances],
    }


# ---------------------------------------------------------------------------
# 9. 保存结果
# ---------------------------------------------------------------------------
def save_results(summary):
    """将关键结果写入 results.json 与 results.txt，方便制作 PPT。"""
    section("9. 保存结果")

    json_path = OUT_DIR / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [文件] {json_path.relative_to(BASE_DIR)}")

    txt_path = OUT_DIR / "results.txt"
    lines = [
        "KNN 鸢尾花分类实验结果",
        "=" * 40,
        f"样本数：{summary['dataset']['n_samples']}",
        f"特征数：{summary['dataset']['n_features']}",
        f"类别：{'、'.join(summary['dataset']['classes'])}",
        f"训练集/测试集：{summary['split']['n_train']} / {summary['split']['n_test']}",
        f"随机种子：{summary['split']['random_state']}",
        "",
        f"最佳 k 值：{summary['best_k']}",
        f"交叉验证准确率：{summary['cv_accuracy']:.4f}",
        f"测试集准确率：{summary['test_accuracy']:.4f}",
        "",
        "各类别指标：",
    ]
    for name in CLASS_CN:
        m = summary["classification_report"][name]
        lines.append(
            f"  {name}: 精确率 {m['precision']:.4f}，"
            f"召回率 {m['recall']:.4f}，F1 {m['f1-score']:.4f}"
        )
    lines += [
        "",
        f"混淆矩阵（行=真实，列=预测）：{summary['confusion_matrix']}",
        "",
        f"未标准化 CV 准确率：{summary['scaling_effect']['raw_cv_accuracy']:.4f}",
        f"标准化后 CV 准确率：{summary['scaling_effect']['scaled_cv_accuracy']:.4f}",
        "",
        f"最近邻演示：{summary['neighbor_demo']['true_label']} → "
        f"{summary['neighbor_demo']['pred_label']}，"
        f"投票 {summary['neighbor_demo']['votes']}",
        "",
        "补充实验（Wine 数据集，13 个特征量纲差异大）：",
        f"  未标准化：CV {summary['wine_supplement']['raw_cv_accuracy']:.4f}，"
        f"测试集 {summary['wine_supplement']['raw_test_accuracy']:.4f}",
        f"  标准化后：CV {summary['wine_supplement']['scaled_cv_accuracy']:.4f}，"
        f"测试集 {summary['wine_supplement']['scaled_test_accuracy']:.4f}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [文件] {txt_path.relative_to(BASE_DIR)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("基于 K 近邻（KNN）的鸢尾花分类与可视化分析")
    print("=" * 72)

    df, X, y, iris = load_data()
    counts = explore_data(df, X, y, iris)

    (
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
    ) = prepare_data(X, y)

    baseline_model(X_train_scaled, X_test_scaled, y_train, y_test, k=5)

    best_k, best_cv_acc, _, _ = select_k(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    model, y_pred, test_acc, report, cm = evaluate_final_model(
        X_train_scaled, X_test_scaled, y_train, y_test, best_k
    )

    plot_decision_boundaries(
        X_train, X_test, y_train, y_test, best_k
    )

    raw_cv, scaled_cv = plot_scaling_effect(
        X_train, y_train, X_train_scaled, best_k
    )

    neighbor_demo = plot_nearest_neighbors(
        X_train, X_test, y_train, y_test, best_k
    )

    wine_result = wine_scaling_experiment()

    summary = {
        "dataset": {
            "name": "Iris（鸢尾花）",
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "feature_names": FEATURE_CN,
            "classes": CLASS_CN,
            "class_counts": {k: int(v) for k, v in counts.items()},
        },
        "split": {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "best_k": int(best_k),
        "cv_accuracy": float(best_cv_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "scaling_effect": {
            "raw_cv_accuracy": float(raw_cv),
            "scaled_cv_accuracy": float(scaled_cv),
        },
        "neighbor_demo": neighbor_demo,
        "wine_supplement": wine_result,
    }
    save_results(summary)

    section("完成")
    print(f"主程序共生成 9 张图表，保存在 {FIG_DIR.relative_to(BASE_DIR)}/")
    print("演示时建议顺序：01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 10")
    print("另运行 knn_from_scratch.py 生成图 09（手写实现对比）")

    if ARGS.show:
        plt.show()


if __name__ == "__main__":
    main()
