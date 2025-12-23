import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
import warnings
import os
import matplotlib.font_manager as fm  # 引入字体管理模块

# ===================== 1. 基础配置 =====================
st.set_page_config(
    page_title="IPL 球员生命周期可视化分析系统",
    page_icon="🏏",
    layout="wide"
)

# 消除警告
warnings.filterwarnings('ignore')

# ----------------- 字体设置 (解决中文显示方框问题) -----------------
# 尝试加载本地字体文件 (font.otf 或 font.ttf)
# 请确保你已将字体文件上传到 GitHub 并重命名为 font.otf
font_files = ['font.otf', 'font.ttf', 'simhei.ttf']
font_loaded = False

for font_file in font_files:
    if os.path.exists(font_file):
        try:
            # 1. 加载字体文件
            fm.fontManager.addfont(font_file)
            # 2. 获取字体内部名称
            font_prop = fm.FontProperties(fname=font_file)
            custom_font_name = font_prop.get_name()
            # 3. 设置为全局默认字体
            plt.rcParams['font.family'] = custom_font_name
            font_loaded = True
            break
        except Exception as e:
            print(f"字体加载失败: {e}")

# 如果没找到本地字体，尝试使用系统回退字体
if not font_loaded:
    import platform
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':  # MacOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        # Linux (Streamlit Cloud) 默认没有中文字体，如果走到这里可能会显示方框
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
# ------------------------------------------------------------------

# ===================== 2. 数据预处理逻辑 =====================
@st.cache_data
def load_and_preprocess_data(file_path_or_buffer):
    """
    数据读取与预处理函数
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
        
        # --- 核心预处理步骤 ---
        # 1. 关键列处理
        if 'Player_Name' in df.columns and 'Year' in df.columns:
            df = df.dropna(subset=['Player_Name', 'Year'])
        
        # 2. 异常值标记替换 (No stats -> NaN)
        stats_columns = ['Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
                        'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries', 'Fours', 'Sixes',
                        'Catches_Taken', 'Stumpings', 'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded',
                        'Wickets_Taken', 'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate',
                        'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls']
        
        for col in stats_columns:
            if col in df.columns:
                df[col] = df[col].replace('No stats', np.nan)
                # 转换为数值型 (除了特殊列)
                if col not in ['Best_Bowling_Match', 'Highest_Score']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 确保年份是数值
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # 4. 去重逻辑
        df['核心键'] = df['Player_Name'].astype(str) + '_' + df['Year'].astype(str).fillna('NaN')
        df = df.drop_duplicates(subset=['核心键'], keep='first')
        df.drop('核心键', axis=1, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"数据处理出错: {e}")
        return None

# ===================== 3. 图表绘制函数集 (16个图) =====================

def plot_fig1(df):
    """图1：球员年度总跑位得分分布直方图"""
    valid_runs = df[df['Runs_Scored'].notna()].copy()
    valid_runs['Runs_Scored'] = pd.to_numeric(valid_runs['Runs_Scored'], errors='coerce')
    valid_runs = valid_runs[valid_runs['Runs_Scored'] > 0]
    range_0_150 = len(valid_runs[(valid_runs['Runs_Scored'] >= 0) & (valid_runs['Runs_Scored'] <= 150)])
    range_500_plus = len(valid_runs[valid_runs['Runs_Scored'] >= 500])
    rate_0_150 = round((range_0_150 / len(valid_runs) * 100), 1)
    rate_500_plus = round((range_500_plus / len(valid_runs) * 100), 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(valid_runs['Runs_Scored'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    for i, patch in enumerate(patches):
        if bins[i] >= 0 and bins[i+1] <= 150: patch.set_facecolor('orange')
    ax.text(75, max(n)*0.8, f'0-150分: {rate_0_150}%', ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(700, max(n)*0.5, f'500+分: {rate_500_plus}%', ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.set_title('球员年度总跑位得分分布')
    ax.set_xlabel('总跑位得分'); ax.set_ylabel('球员人数')
    return fig

def plot_fig2(df):
    """图2：三柱门数与投球平均失分数散点图"""
    valid_bowling = df[(df['Wickets_Taken'] > 0) & (df['Bowling_Average'] > 0)].copy()
    corr, _ = pearsonr(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'], alpha=0.6, color='coral')
    ax.text(valid_bowling['Wickets_Taken'].max()*0.7, valid_bowling['Bowling_Average'].max()*0.8, f'Pearson: {round(corr, 2)}', bbox=dict(facecolor='lightblue', alpha=0.8))
    ax.set_title('三柱门数与投球平均失分数关系')
    ax.set_xlabel('三柱门数'); ax.set_ylabel('投球平均失分数')
    return fig

def plot_fig3(df):
    """图3：Virat Kohli 年度表现趋势"""
    kohli_df = df[df['Player_Name'] == 'Virat Kohli'].copy()
    if kohli_df.empty: return plt.figure()
    kohli_df = kohli_df.sort_values('Year')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(kohli_df['Year'], kohli_df['Runs_Scored'], 'b-o', label='得分')
    ax2 = ax1.twinx()
    ax2.plot(kohli_df['Year'], kohli_df['Wickets_Taken'], 'r-s', label='三柱门')
    ax1.set_title('Virat Kohli 表现趋势')
    ax1.set_xlabel('年份'); ax1.set_ylabel('得分', color='b'); ax2.set_ylabel('三柱门', color='r')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    return fig

def plot_fig4(df):
    """图4：不同年份球员击球平均率箱线图"""
    target_years = [2010, 2015, 2020, 2024]
    data = [df[(df['Year'] == y) & (df['Batting_Average'] > 0)]['Batting_Average'] for y in target_years]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=target_years, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('不同年份击球平均率分布')
    return fig

def plot_fig5(df):
    """图5：顶级球员多维度雷达图"""
    target_players = ['Virat Kohli', 'MS Dhoni', 'Suryakumar Yadav']
    indicators = ['Batting_Average', 'Batting_Strike_Rate', 'Wickets_Taken', 'Bowling_Average', 'Catches_Taken']
    player_df = df[df['Player_Name'].isin(target_players)].copy()
    for col in indicators: player_df[col] = pd.to_numeric(player_df[col], errors='coerce').fillna(0)
    best_df = player_df.groupby('Player_Name').apply(lambda x: x.nlargest(1, 'Runs_Scored')).reset_index(drop=True)
    if best_df.empty: return plt.figure()
    
    radar_data = []
    for _, row in best_df.iterrows():
        scores = []
        for col in indicators:
            max_val = df[col].max()
            val = row[col]
            if col == 'Bowling_Average': 
                score = 10 - (val/max_val*10) if max_val>0 else 0
            else:
                score = (val/max_val*10) if max_val>0 else 0
            scores.append(max(0, min(10, score)))
        radar_data.append(scores)
        
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, scores in enumerate(radar_data):
        ax.plot(angles, scores + scores[:1], label=best_df.iloc[i]['Player_Name'])
        ax.fill(angles, scores + scores[:1], alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(indicators)
    ax.set_title('顶级球员雷达图'); ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    return fig

def plot_fig6(df):
    """图6：组合图 (网页版简化提示)"""
    fig = plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, "由于网页空间限制，\n请分别点击其他选项查看各分图详情", ha='center', fontsize=14)
    plt.axis('off')
    return fig

def plot_fig7(df):
    """图7：效率散点图"""
    df_v = df[(df['Year']>=2010)].copy()
    df_v['Eff_Bat'] = df_v['Runs_Scored']/df_v['Matches_Batted']
    df_v['Eff_Bowl'] = df_v['Wickets_Taken']/df_v['Matches_Bowled']
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df_v['Matches_Batted'], df_v['Eff_Bat'], alpha=0.5, label='击球效率')
    ax2 = ax.twinx()
    ax2.scatter(df_v['Matches_Bowled'], df_v['Eff_Bowl'], color='r', alpha=0.5, label='投球效率')
    ax.set_title('参赛场次与效率'); ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    return fig

def plot_fig8(df):
    """图8：得分结构堆叠图"""
    df_v = df[(df['Year']>=2010)].groupby('Year')[['Centuries', 'Half_Centuries', 'Fours', 'Sixes']].sum()
    df_v = df_v.div(df_v.sum(axis=1), axis=0)*100
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(df_v.index, df_v.T, labels=df_v.columns, alpha=0.7)
    ax.legend(); ax.set_title('得分结构年度变化')
    return fig

def plot_fig9(df):
    """图9：平均率区间分布"""
    df_v = df[df['Year']>=2010].copy()
    df_v['Group'] = pd.cut(df_v['Batting_Average'], bins=[0,10,20,30,40,50,100])
    stats = df_v.groupby('Group')['Runs_Scored'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stats.index.astype(str), stats.values, 'o-')
    ax.set_title('平均率区间与平均得分'); ax.set_ylabel('平均得分')
    return fig

def plot_fig10(df):
    """图10：TOP5球员趋势"""
    top5 = df.groupby('Player_Name')['Runs_Scored'].sum().nlargest(5).index
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in top5:
        data = df[df['Player_Name']==p].groupby('Year')['Runs_Scored'].sum()
        ax.plot(data.index, data.values, marker='o', label=p)
    ax.legend(); ax.set_title('Top 5 球员得分趋势')
    return fig

def plot_fig11(df):
    """图11：投球效率热力图"""
    data = df[(df['Bowling_Average']>0) & (df['Wickets_Taken']>0)]
    fig, ax = plt.subplots(figsize=(10, 6))
    h = ax.hist2d(data['Bowling_Average'], data['Wickets_Taken'], bins=20, cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax)
    ax.set_title('投球效率热力图'); ax.set_xlabel('失分数'); ax.set_ylabel('三柱门')
    return fig

def plot_fig12(df):
    """图12：参赛年份分布"""
    data = df[df['Year']>=2008].groupby('Year')['Player_Name'].nunique()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data.index, data.values, color='skyblue')
    ax.set_title('参赛球员数量分布'); ax.set_xlabel('人数')
    return fig

def plot_fig13(df):
    """图13：稳定性分析 (小提琴图)"""
    df_v = df[df['Batting_Average']>0].groupby('Player_Name')['Batting_Average'].agg(['mean','std']).dropna()
    df_v['CV'] = df_v['std']/df_v['mean']
    df_v = df_v[df_v['CV']<=2]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.violinplot(df_v['CV'], showmedians=True)
    ax.set_title('球员表现稳定性 (CV)'); ax.set_ylabel('变异系数')
    return fig

def plot_fig14(df):
    """图14：投手象限分析"""
    df_p = df[(df['Year']>=2020) & (df['Balls_Bowled']>0)].groupby('Player_Name').mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_p['Economy_Rate'], df_p['Wickets_Taken'], alpha=0.5)
    ax.set_title('投手象限分析'); ax.set_xlabel('经济率'); ax.set_ylabel('场均三柱门')
    return fig

def plot_fig15(df):
    """图15：球员类型分布"""
    df_v = df[(df['Year']>=2010)].copy()
    def get_type(row):
        if row['Batting_Average']>=25: return '击球手'
        if row['Wickets_Taken']>=5: return '投手'
        return '其他'
    df_v['Type'] = df_v.apply(get_type, axis=1)
    data = df_v.groupby(['Year', 'Type']).size().unstack().fillna(0)
    data = data.div(data.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    data.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('球员类型分布'); ax.set_ylabel('占比')
    return fig

def plot_fig16(df):
    """图16：接球与综合表现"""
    df_v = df[df['Year']>=2018].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_v['Catches_Taken'], df_v['Runs_Scored'], alpha=0.5)
    ax.set_title('接球与得分相关性'); ax.set_xlabel('接球数'); ax.set_ylabel('得分')
    return fig

# ===================== 4. Streamlit 页面布局 =====================

st.title("🏏 IPL 顶级球员生命周期与表现可视化系统")

# 定义数据文件名
DEFAULT_FILE = "data.csv"
ALT_FILE = "6-球员生命周期_预处理后.csv"

# 初始化
df = None
loaded_msg = ""

# 1. 尝试自动加载
if os.path.exists(DEFAULT_FILE):
    df = load_and_preprocess_data(DEFAULT_FILE)
    loaded_msg = f"已自动加载本地数据 ({DEFAULT_FILE})"
elif os.path.exists(ALT_FILE):
    df = load_and_preprocess_data(ALT_FILE)
    loaded_msg = f"已自动加载本地数据 ({ALT_FILE})"

# 侧边栏
st.sidebar.header("数据与设置")

if df is not None:
    st.sidebar.success(f"✅ {loaded_msg}")
    st.sidebar.info(f"包含 {len(df)} 条记录")
    
    if st.sidebar.checkbox("上传新文件覆盖"):
        uploaded_file = st.sidebar.file_uploader("上传 CSV", type=['csv'])
        if uploaded_file is not None:
            df = load_and_preprocess_data(uploaded_file)
            st.sidebar.success("已切换为上传的数据")
else:
    st.sidebar.warning("⚠️ 未检测到本地 data.csv")
    uploaded_file = st.sidebar.file_uploader("请上传 CSV 数据文件", type=['csv'])
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)

# 主逻辑
if df is not None:
    st.markdown("---")
    
    # 侧边栏：图表选择
    st.sidebar.header("📊 图表导航")
    category = st.sidebar.selectbox(
        "选择分析维度",
        ["数据总览", "击球表现分析", "投球表现分析", "综合与相关性分析", "球员特写"]
    )
    
    chart_map = {
        "数据总览": {
            "图1: 球员年度得分分布": plot_fig1,
            "图12: 参赛球员年份分布": plot_fig12,
            "图15: 球员类型年度分布": plot_fig15
        },
        "击球表现分析": {
            "图4: 击球平均率箱线图": plot_fig4,
            "图8: 得分结构堆叠图": plot_fig8,
            "图9: 平均率区间球员分布": plot_fig9,
            "图10: TOP5球员得分趋势": plot_fig10,
            "图13: 参赛年限与稳定性": plot_fig13
        },
        "投球表现分析": {
            "图2: 三柱门数 vs 失分数": plot_fig2,
            "图11: 投球效率热力图": plot_fig11,
            "图14: 投手经济率象限分析": plot_fig14
        },
        "综合与相关性分析": {
            "图6: 综合分析组合图": plot_fig6,
            "图7: 参赛场次与效率": plot_fig7,
            "图16: 接球能力与综合表现": plot_fig16
        },
        "球员特写": {
            "图3: Virat Kohli 年度趋势": plot_fig3,
            "图5: 顶级球员雷达图": plot_fig5
        }
    }
    
    selected_chart_name = st.sidebar.radio("选择图表", list(chart_map[category].keys()))
    plot_func = chart_map[category][selected_chart_name]
    
    # 主界面显示
    st.subheader(f"📈 {selected_chart_name}")
    
    try:
        fig = plot_func(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"图表生成失败: {e}")
        st.write("可能原因：数据列名不匹配或缺少关键字段")

    # 底部数据预览
    with st.expander("🔍 查看源数据"):
        st.dataframe(df.head())

else:
    st.info("👋 请上传数据以开始分析。")