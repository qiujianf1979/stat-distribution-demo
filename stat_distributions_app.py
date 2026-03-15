import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t, f
import streamlit as st

# 让 matplotlib 在 Streamlit 中使用非交互式后端
plt.switch_backend("agg")

np.random.seed(42)

st.set_page_config(
    page_title="三大分布动态演示：卡方 / t / F",
    layout="centered"
)

st.title("数理统计三大分布的动态演示：卡方 · t · F")
st.markdown(
    """
**教学思路**：  
通过“先模拟检验统计量，再看它的分布形状”，让学生理解：

- 卡方分布：来自 **方差检验统计量**  
- t 分布：来自 **小样本均值检验统计量**  
- F 分布：来自 **方差比 / ANOVA F 统计量**

在侧边栏选择分布类型和参数，观察图形如何变化。
"""
)

# ---------------------------
# 侧边栏：分布类型 & 参数设置
# ---------------------------
st.sidebar.header("参数设置")

dist_type = st.sidebar.selectbox(
    "选择分布类型",
    [
        "卡方分布（方差检验）",
        "t 分布（均值检验）",
        "F 分布：两个方差",
        "F 分布：ANOVA"
    ]
)

N_sim = st.sidebar.slider(
    "模拟次数 N_sim（越大越平滑，但会稍慢）",
    min_value=1000,
    max_value=15000,
    step=1000,
    value=4000
)

# 公共参数（会按需要使用）
n = st.sidebar.slider("样本量 n（或 n1）", 5, 80, 15, 1)
n2 = st.sidebar.slider("样本量 n2（F: 第二个样本）", 5, 80, 20, 1)
k = st.sidebar.slider("组数 k（ANOVA）", 2, 8, 3, 1)
n_per_group = st.sidebar.slider("每组样本量（ANOVA）", 4, 60, 10, 1)

# ---------------------------
# 各分布的模拟与绘图函数
# ---------------------------

def plot_chi_square(n, N_sim):
    sigma0 = 1.0
    samples = np.random.normal(0, sigma0, size=(N_sim, n))
    s2 = samples.var(axis=1, ddof=1)
    chi_stats = (n - 1) * s2 / (sigma0 ** 2)

    df = n - 1
    x = np.linspace(0.001, chi2.ppf(0.999, df), 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        chi_stats,
        bins=40,
        density=True,
        alpha=0.6,
        label="模拟统计量分布"
    )
    ax.plot(x, chi2.pdf(x, df), "r-", lw=2, label=f"理论卡方分布 (df={df})")
    ax.set_title("卡方分布的来源：方差检验统计量的分布")
    ax.set_xlabel(r"$(n-1) S^2 / \sigma_0^2$")
    ax.set_ylabel("密度")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_t(n, N_sim):
    mu0 = 70
    sigma_true = 10
    samples = np.random.normal(mu0, sigma_true, size=(N_sim, n))
    xbar = samples.mean(axis=1)
    s = samples.std(axis=1, ddof=1)
    t_stats = (xbar - mu0) / (s / np.sqrt(n))

    df = n - 1
    x = np.linspace(t.ppf(0.001, df), t.ppf(0.999, df), 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        t_stats,
        bins=40,
        density=True,
        alpha=0.6,
        label="模拟 t 统计量分布"
    )
    ax.plot(x, t.pdf(x, df), "r-", lw=2, label=f"理论 t 分布 (df={df})")
    ax.set_title("t 分布的来源：小样本均值检验统计量的分布")
    ax.set_xlabel(r"$T = (\bar{X} - \mu_0) / (S / \sqrt{n})$")
    ax.set_ylabel("密度")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_F_two_var(n1, n2, N_sim):
    sigma = 2.0
    s1 = np.random.normal(0, sigma, size=(N_sim, n1))
    s2 = np.random.normal(0, sigma, size=(N_sim, n2))
    s1sq = s1.var(axis=1, ddof=1)
    s2sq = s2.var(axis=1, ddof=1)
    F_stats = s1sq / s2sq

    df1, df2 = n1 - 1, n2 - 1
    x_max = f.ppf(0.995, df1, df2)
    x = np.linspace(0.001, x_max, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        F_stats,
        bins=40,
        density=True,
        alpha=0.6,
        label="模拟 F 统计量分布"
    )
    ax.plot(x, f.pdf(x, df1, df2), "r-", lw=2,
            label=f"理论 F 分布 (df1={df1}, df2={df2})")
    ax.set_title("F 分布的来源：两个样本方差比的分布")
    ax.set_xlabel(r"$F = S_1^2 / S_2^2$")
    ax.set_ylabel("密度")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_F_anova(k, n_per_group, N_sim):
    mu = 75
    sigma = 8
    F_values = []

    for _ in range(N_sim):
        groups = [np.random.normal(mu, sigma, size=n_per_group) for _ in range(k)]
        all_data = np.concatenate(groups)
        grand_mean = all_data.mean()

        SSB = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        SSW = sum(((g - g.mean()) ** 2).sum() for g in groups)

        df_between = k - 1
        df_within = k * n_per_group - k

        MSB = SSB / df_between
        MSW = SSW / df_within

        F_values.append(MSB / MSW)

    F_values = np.array(F_values)
    df1, df2 = k - 1, k * n_per_group - k
    x_max = f.ppf(0.995, df1, df2)
    x = np.linspace(0.001, x_max, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        F_values,
        bins=40,
        density=True,
        alpha=0.6,
        label="模拟 ANOVA F 统计量分布"
    )
    ax.plot(x, f.pdf(x, df1, df2), "r-", lw=2,
            label=f"理论 F 分布 (df1={df1}, df2={df2})")
    ax.set_title("F 分布的来源：单因素方差分析 F 统计量的分布")
    ax.set_xlabel("F = MSB / MSW")
    ax.set_ylabel("密度")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

# ---------------------------
# 主区域：根据选择绘图 + 文本解释
# ---------------------------

if dist_type == "卡方分布（方差检验）":
    st.subheader("卡方分布：用来做“方差是否改变”的检验")
    st.markdown(
        """
**故事场景**：  
一台机器原来生产的零件，理论上标准差为 \\(\\sigma_0\\)。  
现在抽取 \\(n\\) 个零件测量，样本方差为 \\(S^2\\)，检验“方差是否变化”。

在原假设 \\(H_0: \\sigma^2 = \\sigma_0^2\\) 成立时，
统计量  
\\[
\\chi^2 = \\frac{(n-1)S^2}{\\sigma_0^2}
\\]
服从卡方分布 \\(\\chi^2(n-1)\\)。

图中蓝色直方图是通过模拟样本得到的 \\(\\chi^2\\) 分布，
红色曲线是理论卡方分布。
"""
    )
    fig = plot_chi_square(n, N_sim)
    st.pyplot(fig)

elif dist_type == "t 分布（均值检验）":
    st.subheader("t 分布：小样本条件下比较均值")
    st.markdown(
        """
**故事场景**：  
原来某课程平均成绩为 \\(\\mu_0\\)，方差未知。  
改革后只抽到 \\(n\\) 个学生样本，想检验“平均成绩是否有变化”。

在总体正态、方差未知条件下，若 \\(H_0: \\mu = \\mu_0\\) 成立，

\\[
T = \\frac{\\bar{X} - \\mu_0}{S / \\sqrt{n}} \\sim t(n-1)
\\]

图中蓝色直方图是通过模拟样本计算出来的 \\(T\\) 值分布，
红色曲线是理论 t 分布。
**拖动 n，体会小样本时尾部更“厚”，n 变大时逐渐接近标准正态。**
"""
    )
    fig = plot_t(n, N_sim)
    st.pyplot(fig)

elif dist_type == "F 分布：两个方差":
    st.subheader("F 分布：比较两个总体方差（波动大小）")
    st.markdown(
        """
**故事场景**：  
两个工厂都生产同一种零件，我们抽取两组样本，样本量分别为 \\(n_1, n_2\\)，
样本方差分别为 \\(S_1^2, S_2^2\\)，关心“哪个工厂波动更大”。

在 \\(H_0: \\sigma_1^2 = \\sigma_2^2\\) 成立且总体正态时，

\\[
F = \\frac{S_1^2}{S_2^2} \\sim F(n_1-1, n_2-1)
\\]

图中蓝色直方图是模拟得到的方差比，红色曲线是理论 F 分布。
**注意：F 分布永远在 0 右侧，并且是右偏的。**
"""
    )
    fig = plot_F_two_var(n, n2, N_sim)
    st.pyplot(fig)

elif dist_type == "F 分布：ANOVA":
    st.subheader("F 分布：单因素方差分析（ANOVA）中的 F 统计量")
    st.markdown(
        """
**故事场景**：  
比较 \\(k\\) 种教学方法（或 \\(k\\) 个班级）的平均成绩是否相同。  
每组抽取 \\(n\\) 个学生。

在原假设 \\(H_0: \\mu_1 = \\mu_2 = \\cdots = \\mu_k\\) 成立，且总体正态、方差相同条件下，

单因素 ANOVA 中的 F 统计量：

\\[
F = \\frac{\\text{组间均方 MSB}}{\\text{组内均方 MSW}} \\sim F(k-1, k(n-1))
\\]

图中蓝色直方图是通过模拟数据计算 F 的经验分布，
红色曲线是理论 F 分布。
**拖动“组数 k”和“每组样本量”，体会自由度变化对 F 分布形状的影响。**
"""
    )
    fig = plot_F_anova(k, n_per_group, N_sim)
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    """
**课堂使用建议（简要）**：

- 先让学生猜：如果原假设成立，统计量大致会落在哪一块？  
- 拖动滑块改变样本量 / 组数，让学生观察分布形状如何变化。  
- 强调一句：**这些分布不是凭空来的，而是“在原假设为真时，检验统计量的分布”**。
"""
)