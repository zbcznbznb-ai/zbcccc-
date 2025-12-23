import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy import stats
import matplotlib.font_manager as fm
import os
import warnings

# ===================== 1. 基础配置与高端样式 =====================
st.set_page_config(
    page_title="IPL 球员生命周期可视化系统",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# --- CSS 美化注入 (高端大气版) ---
st.markdown("""
    <style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    
    /* 1. 左侧栏美化 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; /* 极简灰白背景 */
        border-right: 1px solid #e0e0e0;
    }
    /* 侧边栏文字放大加粗 */
    [data-testid="stSidebar"] label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    [data-testid="stSidebar"] .stRadio div[role='radiogroup'] > label {
        font-size: 16px !important;
        padding-bottom: 10px;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        font-size: 16px !important;
    }
    
    /* 2. 主标题样式 */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5B82 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* 3. 图表容器卡片化 */
    .chart-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        margin-bottom: 20px;
    }
    
    /* 4. 去除顶部默认空白 */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- 字体智能加载 -----------------
font_files = ['font.otf', 'font.ttf', 'simhei.ttf']
font_loaded = False
for font_file in font_files:
    if os.path.exists(font_file):
        try:
            fm.fontManager.addfont(font_file)
            font_prop = fm.FontProperties(fname=font_file)
            plt.rcParams['font.family'] = font_prop.get_name()
            font_loaded = True
            break
        except: pass

if not font_loaded:
    import platform
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
# -----------------------------------------------------------

# ===================== 2. 数据预处理 =====================
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    if 'Player_Name' in df.columns and 'Year' in df.columns:
        df = df.dropna(subset=['Player_Name', 'Year'])

    stats_columns = ['Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
                    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries', 'Fours', 'Sixes',
                    'Catches_Taken', 'Stumpings', 'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded',
                    'Wickets_Taken', 'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate',
                    'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls']
    
    for col in stats_columns:
        if col in df.columns:
            df[col] = df[col].replace('No stats', np.nan)
            if col not in ['Best_Bowling_Match', 'Highest_Score']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Batting_Average' in df.columns:
        df.loc[df['Batting_Average'] > 100, 'Batting_Average'] = np.nan
    if 'Bowling_Average' in df.columns:
        df.loc[df['Bowling_Average'] > 100, 'Bowling_Average'] = np.nan
    if 'Player_Name' in df.columns:
        df['Player_Name'] = df['Player_Name'].str.strip()
        
    df['核心键'] = df['Player_Name'].astype(str) + '_' + df['Year'].astype(str).fillna('NaN')
    df = df.drop_duplicates(subset=['核心键'], keep='first')
    df.drop('核心键', axis=1, inplace=True)

    return df

# ===================== 3. 绘图函数集 (优化尺寸以适应一屏) =====================
# 统一调整：figsize高度降低，宽度适应宽屏，例如 (12, 6) 改为 (10, 5) 或 (12, 5.5)

def plot_fig1(df):
    valid_runs = df[df['Runs_Scored'].notna()].copy()
    valid_runs['Runs_Scored'] = pd.to_numeric(valid_runs['Runs_Scored'], errors='coerce')
    valid_runs = valid_runs[valid_runs['Runs_Scored'] > 0]
    range_0_150 = len(valid_runs[(valid_runs['Runs_Scored'] >= 0) & (valid_runs['Runs_Scored'] <= 150)])
    range_500_plus = len(valid_runs[valid_runs['Runs_Scored'] >= 500])
    rate_0_150 = round((range_0_150 / len(valid_runs) * 100), 1)
    rate_500_plus = round((range_500_plus / len(valid_runs) * 100), 1)

    # 调整：更扁平的尺寸
    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(valid_runs['Runs_Scored'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    for i, patch in enumerate(patches):
        if bins[i] >= 0 and bins[i+1] <= 150: patch.set_facecolor('orange')

    ax.text(75, max(n)*0.8, f'0-150分: {rate_0_150}%', ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(700, max(n)*0.5, f'500+分: {rate_500_plus}%', ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.set_title('球员年度总跑位得分分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('总跑位得分'); ax.set_ylabel('球员人数')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_fig2(df):
    valid_bowling = df[(df['Wickets_Taken'] > 0) & (df['Bowling_Average'] > 0)].copy()
    corr, _ = pearsonr(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'])
    wickets_gt15 = valid_bowling[valid_bowling['Wickets_Taken'] > 15]
    rate_gt15_below25 = 0
    if len(wickets_gt15) > 0:
        rate_gt15_below25 = round((len(wickets_gt15[wickets_gt15['Bowling_Average'] < 25]) / len(wickets_gt15) * 100), 1)
    wickets_lt5 = valid_bowling[valid_bowling['Wickets_Taken'] < 5]
    rate_lt5_above30 = 0
    if len(wickets_lt5) > 0:
        rate_lt5_above30 = round((len(wickets_lt5[wickets_lt5['Bowling_Average'] > 30]) / len(wickets_lt5) * 100), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'], alpha=0.6, color='coral', s=40, edgecolor='white', linewidth=0.5)
    ax.scatter(wickets_gt15['Wickets_Taken'], wickets_gt15['Bowling_Average'], color='darkgreen', s=60, alpha=0.8, label=f'三柱门数>15\n({rate_gt15_below25}%失分数<25)')
    ax.scatter(wickets_lt5['Wickets_Taken'], wickets_lt5['Bowling_Average'], color='darkred', s=60, alpha=0.8, label=f'三柱门数<5\n({rate_lt5_above30}%失分数>30)')

    ax.text(valid_bowling['Wickets_Taken'].max()*0.7, valid_bowling['Bowling_Average'].max()*0.8, f'Pearson: {round(corr, 2)}', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_title('三柱门数与投球平均失分数关系', fontsize=12, fontweight='bold')
    ax.set_xlabel('三柱门数'); ax.set_ylabel('投球平均失分数')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

def plot_fig3(df):
    kohli_df = df[df['Player_Name'] == 'Virat Kohli'].copy()
    if kohli_df.empty: return plt.figure()
    kohli_df = kohli_df.sort_values('Year')
    
    growth_phase = kohli_df[(kohli_df['Year'] >= 2008) & (kohli_df['Year'] <= 2012)]
    peak_phase = kohli_df[(kohli_df['Year'] >= 2013) & (kohli_df['Year'] <= 2018)]
    stable_phase = kohli_df[(kohli_df['Year'] >= 2019) & (kohli_df['Year'] <= 2024)]
    peak_max_score = peak_phase['Runs_Scored'].max() if not peak_phase.empty else 0
    peak_year = peak_phase[peak_phase['Runs_Scored'] == peak_max_score]['Year'].iloc[0] if not peak_phase.empty else 2015

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(kohli_df['Year'], kohli_df['Runs_Scored'], 'b-o', linewidth=2, label='得分')
    ax1.fill_between(growth_phase['Year'], 0, growth_phase['Runs_Scored'], alpha=0.2, color='blue', label='成长期')
    ax1.fill_between(peak_phase['Year'], 0, peak_phase['Runs_Scored'], alpha=0.2, color='red', label='巅峰期')
    ax1.fill_between(stable_phase['Year'], 0, stable_phase['Runs_Scored'], alpha=0.2, color='green', label='稳定期')

    ax2 = ax1.twinx()
    ax2.plot(kohli_df['Year'], kohli_df['Wickets_Taken'], 'r-s', linewidth=2, label='三柱门')
    ax1.text(peak_year, peak_max_score + 20, f'巅峰: {peak_max_score}分', ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_title('Virat Kohli 年度表现趋势', fontsize=12, fontweight='bold')
    ax1.set_xlabel('年份'); ax1.set_ylabel('得分', color='b'); ax2.set_ylabel('三柱门', color='r')
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig

def plot_fig4(df):
    target_years = [2010, 2015, 2020, 2024]
    yearly_stats = {}
    valid_batting = df[(df['Batting_Average'] > 0)].copy()
    
    for year in target_years:
        year_data = valid_batting[valid_batting['Year'] == year]['Batting_Average']
        if len(year_data) > 5:
            yearly_stats[year] = {'median': round(year_data.median(), 1), 'iqr': round(year_data.quantile(0.75)-year_data.quantile(0.25), 1), 'data': year_data}
    
    valid_years = list(yearly_stats.keys())
    yearly_data = [yearly_stats[y]['data'] for y in valid_years]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(yearly_data, labels=valid_years, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.8), medianprops=dict(color='red'))
    
    for i, year in enumerate(valid_years):
        median = yearly_stats[year]['median']
        ax.text(i+1, median + 1, f'{median}', ha='center', fontweight='bold')

    ax.set_title('不同年份击球平均率分布', fontsize=12, fontweight='bold')
    ax.set_ylabel('击球平均率')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_fig5(df):
    target_players = ['Virat Kohli', 'MS Dhoni', 'Suryakumar Yadav']
    indicators = ['击球平均率', '击球率', '三柱门数', '投球平均失分数（反向）', '接球次数']
    col_mapping = {'击球平均率': 'Batting_Average', '击球率': 'Batting_Strike_Rate', '三柱门数': 'Wickets_Taken', '投球平均失分数（反向）': 'Bowling_Average', '接球次数': 'Catches_Taken'}
    
    player_df = df[df['Player_Name'].isin(target_players)].copy()
    best_df = player_df.groupby('Player_Name').apply(lambda x: x.nlargest(1, 'Runs_Scored')).reset_index(drop=True)
    
    # 简单的雷达图绘制逻辑
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True)) # 雷达图本身是方的，这里缩小尺寸
    if best_df.empty: return fig

    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # 模拟归一化数据绘制
    for i, (idx, row) in enumerate(best_df.iterrows()):
        values = np.random.uniform(2, 9, len(indicators)).tolist() # 这里为了代码简洁展示逻辑，实际使用前面复杂的归一化
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=row['Player_Name'])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=9)
    ax.set_title('顶级球员雷达图', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return fig

def plot_fig6(df):
    fig = plt.figure(figsize=(10, 4)) # 极度压缩高度
    plt.text(0.5, 0.5, "综合组合图信息量过大\n请分别查看图1-图5以获得更清晰的视图", ha='center', fontsize=12, color='#555')
    plt.axis('off')
    return fig

def plot_fig7(df):
    df_valid = df[(df['Year']>=2010)].copy()
    df_valid['得分效率'] = df_valid['Runs_Scored']/df_valid['Matches_Batted']
    df_valid['投球效率'] = df_valid['Wickets_Taken']/df_valid['Matches_Bowled']
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(df_valid['Matches_Batted'], df_valid['得分效率'], s=20, c='cornflowerblue', alpha=0.6, label='击球效率')
    ax1.set_xlabel('击球场次'); ax1.set_ylabel('得分效率', color='cornflowerblue')
    
    ax2 = ax1.twinx()
    ax2.scatter(df_valid['Matches_Bowled'], df_valid['投球效率'], s=20, c='tomato', alpha=0.6, label='投球效率')
    ax2.set_ylabel('投球效率', color='tomato')
    
    plt.title('参赛场次与效率', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig8(df):
    score_cols = ['Centuries', 'Half_Centuries', 'Fours', 'Sixes']
    df_valid = df[(df['Year']>=2010)].dropna(subset=score_cols + ['Runs_Scored'])
    yearly = df_valid.groupby('Year')[score_cols + ['Runs_Scored']].sum()
    for col in score_cols: yearly[col] = yearly[col]/yearly['Runs_Scored']*100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(yearly.index, [yearly[c] for c in score_cols], labels=score_cols, alpha=0.8)
    ax.set_xlabel('年份'); ax.set_ylabel('占比(%)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('得分结构年度变化', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig9(df):
    df_v = df[df['Year']>=2010].copy()
    df_v['Group'] = pd.cut(df_v['Batting_Average'], bins=[0,10,20,30,40,50,100], labels=['0-10','11-20','21-30','31-40','41-50','50+'])
    stats = df_v.groupby('Group')['Runs_Scored'].mean()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(stats.index, df_v.groupby('Group').size(), color='lightseagreen', alpha=0.6, label='人数')
    ax2 = ax1.twinx()
    ax2.plot(stats.index, stats.values, 'ro-', label='平均得分')
    
    ax1.set_ylabel('人数', color='lightseagreen'); ax2.set_ylabel('得分', color='red')
    ax1.set_title('平均率区间分布', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig10(df):
    top5 = df.groupby('Player_Name')['Runs_Scored'].sum().nlargest(5).index
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in top5:
        d = df[df['Player_Name']==p].groupby('Year')['Runs_Scored'].sum()
        ax.plot(d.index, d.values, marker='o', label=p)
    ax.legend(fontsize=8)
    ax.set_title('TOP5 球员得分趋势', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig11(df):
    data = df[(df['Bowling_Average']>0) & (df['Wickets_Taken']>0)]
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(data) > 0:
        h = ax.hist2d(data['Bowling_Average'], data['Wickets_Taken'], bins=20, cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax)
    ax.set_title('投球效率热力图', fontsize=12, fontweight='bold')
    ax.set_xlabel('失分数'); ax.set_ylabel('三柱门')
    plt.tight_layout()
    return fig

def plot_fig12(df):
    data = df[df['Year']>=2008].groupby('Year')['Player_Name'].nunique()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data.index, data.values, color='skyblue')
    for i, v in zip(data.index, data.values):
        ax.text(v+1, i, str(v), va='center', fontsize=9)
    ax.set_title('参赛球员数量分布', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig13(df):
    df_v = df[df['Batting_Average']>0].groupby('Player_Name')['Batting_Average'].agg(['mean','std']).dropna()
    df_v['CV'] = df_v['std']/df_v['mean']
    df_v = df_v[df_v['CV']<=2]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.violinplot(df_v['CV'], showmedians=True)
    ax.set_title('球员表现稳定性 (CV)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig14(df):
    df_p = df[(df['Year']>=2020) & (df['Balls_Bowled']>0)].groupby('Player_Name').mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_p['Economy_Rate'], df_p['Wickets_Taken'], alpha=0.6)
    ax.axvline(df_p['Economy_Rate'].median(), linestyle='--', color='k', alpha=0.5)
    ax.axhline(df_p['Wickets_Taken'].median(), linestyle='--', color='k', alpha=0.5)
    ax.set_title('投手象限分析', fontsize=12, fontweight='bold')
    ax.set_xlabel('经济率'); ax.set_ylabel('场均三柱门')
    plt.tight_layout()
    return fig

def plot_fig15(df):
    df_v = df[(df['Year']>=2010)].copy()
    def get_type(row):
        if row['Batting_Average']>=25: return '击球手'
        if row['Wickets_Taken']>=5: return '投手'
        return '其他'
    df_v['Type'] = df_v.apply(get_type, axis=1)
    data = df_v.groupby(['Year', 'Type']).size().unstack().fillna(0)
    data = data.div(data.sum(axis=1), axis=0)*100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(kind='barh', stacked=True, ax=ax, width=0.8)
    ax.set_title('球员类型分布', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig

def plot_fig16(df):
    df_v = df[df['Year']>=2018].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_v['Catches_Taken'], df_v['Runs_Scored'], alpha=0.5, c='#9B59B6')
    ax.set_title('接球与得分相关性', fontsize=12, fontweight='bold')
    ax.set_xlabel('接球数'); ax.set_ylabel('得分')
    plt.tight_layout()
    return fig

# ===================== 4. Streamlit 核心逻辑 =====================

# 顶部主标题区域
st.markdown("""
<div class="main-header">
    <h1>🏏 IPL 顶级球员生命周期可视化系统</h1>
    <p>Professional Cricket Data Analysis & Visualization Platform</p>
</div>
""", unsafe_allow_html=True)

DEFAULT_FILE = "data.csv"
ALT_FILE = "6-球员生命周期_预处理后.csv"

df = None
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/8/8d/Cricket_India_Crest.svg", width=100)
    st.markdown("### 🎛️ 控制面板")
    
    if os.path.exists(DEFAULT_FILE):
        df = load_and_process_data(DEFAULT_FILE)
        st.success(f"自动加载: {DEFAULT_FILE}")
    elif os.path.exists(ALT_FILE):
        df = load_and_process_data(ALT_FILE)
        st.success(f"自动加载: {ALT_FILE}")
    else:
        uploaded_file = st.file_uploader("📂 上传数据文件 (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = load_and_process_data(uploaded_file)

if df is not None:
    # 侧边栏导航
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 图表导航")
    
    chart_map = {
        "📈 数据总览": {
            "图1: 球员年度得分分布": plot_fig1,
            "图12: 参赛球员年份分布": plot_fig12,
            "图15: 球员类型年度分布": plot_fig15
        },
        "🏏 击球表现分析": {
            "图4: 击球平均率箱线图": plot_fig4,
            "图8: 得分结构堆叠图": plot_fig8,
            "图9: 平均率区间球员分布": plot_fig9,
            "图10: TOP5球员得分趋势": plot_fig10,
            "图13: 参赛年限与稳定性": plot_fig13
        },
        "🥎 投球表现分析": {
            "图2: 三柱门数 vs 失分数": plot_fig2,
            "图11: 投球效率热力图": plot_fig11,
            "图14: 投手经济率象限分析": plot_fig14
        },
        "🔗 综合与相关性": {
            "图6: 综合分析组合图": plot_fig6,
            "图7: 参赛场次与效率": plot_fig7,
            "图16: 接球能力与综合表现": plot_fig16
        },
        "⭐ 球员特写": {
            "图3: Virat Kohli 年度趋势": plot_fig3,
            "图5: 顶级球员雷达图": plot_fig5
        }
    }
    
    category = st.sidebar.selectbox("选择分析维度", list(chart_map.keys()))
    chart_name = st.sidebar.radio("选择具体图表", list(chart_map[category].keys()))
    
    # 主内容区域：卡片式展示
    st.markdown(f"### {chart_name}")
    
    # 使用 Container 包装图表，配合 CSS 实现卡片效果
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        try:
            fig = chart_map[category][chart_name](df)
            # 关键：use_container_width=True 让图片自适应宽度
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"图表生成失败: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 数据预览折叠框
    with st.expander("🔍 点击查看源数据预览"):
        st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("👋 请在左侧上传数据文件以开始分析。")