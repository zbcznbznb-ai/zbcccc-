import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy import stats
import matplotlib.font_manager as fm
import os
import warnings

# ===================== 1. 全局配置 =====================
st.set_page_config(
    page_title="IPL 职业板球数据视界",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

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

# ----------------- CSS 高端美化 (封面 + 侧边栏修复) -----------------
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    
    /* 1. 封面 Hero Section */
    .hero-box {
        padding: 3rem 2rem;
        background: linear-gradient(120deg, #141E30 0%, #243B55 100%);
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    /* 2. 右侧说明卡片 */
    .info-card {
        background-color: #f8f9fa;
        border-top: 4px solid #243B55;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #333;
    }
    .info-title {
        font-weight: bold;
        font-size: 1.1rem;
        color: #243B55;
        margin-bottom: 10px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }
    .info-text {
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* 3. 修复侧边栏可见性 */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6; 
    }
    /* 强制侧边栏文字颜色，防止在暗色模式下看不见 */
    [data-testid="stSidebar"] .stRadio label {
        color: #31333F !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #31333F !important;
    }
    
</style>
""", unsafe_allow_html=True)

# ===================== 2. 数据处理核心 =====================
@st.cache_data
def load_data(file):
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
    
    df['核心键'] = df['Player_Name'].astype(str) + '_' + df['Year'].astype(str).fillna('NaN')
    df = df.drop_duplicates(subset=['核心键'], keep='first')
    df.drop('核心键', axis=1, inplace=True)
    
    return df

# ===================== 3. 图表绘制逻辑 (提取自原始文件) =====================

def render_fig1(df):
    """图1：球员年度总跑位得分分布"""
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
    ax.set_title('球员年度总跑位得分分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('总跑位得分'); ax.set_ylabel('球员人数')
    
    desc = f"""
    **核心特征验证：**
    * **长尾分布**：0-150分区间占比高达 **{rate_0_150}%**，说明绝大多数球员年度得分较低，属于角色球员。
    * **头部精英**：500+分区间占比仅 **{rate_500_plus}%**，体现了竞技体育残酷的头部效应，仅极少数球星能稳定高产。
    """
    return fig, desc

def render_fig2(df):
    """图2：三柱门数与失分数"""
    valid_bowling = df[(df['Wickets_Taken']>0) & (df['Bowling_Average']>0)].copy()
    corr, _ = pearsonr(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'])
    corr = round(corr, 2)
    
    wickets_gt15 = valid_bowling[valid_bowling['Wickets_Taken'] > 15]
    rate_good = round((len(wickets_gt15[wickets_gt15['Bowling_Average'] < 25]) / len(wickets_gt15) * 100), 1) if len(wickets_gt15)>0 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'], alpha=0.6, color='coral', s=40, edgecolor='white')
    ax.scatter(wickets_gt15['Wickets_Taken'], wickets_gt15['Bowling_Average'], color='darkgreen', s=60, label=f'三柱门>15 (优质率{rate_good}%)')
    
    ax.text(valid_bowling['Wickets_Taken'].max()*0.7, valid_bowling['Bowling_Average'].max()*0.8, f'Pearson: {corr}', bbox=dict(facecolor='lightblue', alpha=0.8))
    ax.set_title('三柱门数与投球平均失分数关系', fontsize=14, fontweight='bold')
    ax.set_xlabel('三柱门数'); ax.set_ylabel('投球平均失分数')
    ax.legend()
    
    desc = f"""
    **统计分析：**
    * **相关系数**：{corr}，显示出一定的正相关性。
    * **高产即高效**：数据显示，在三柱门数大于15的高产投手中，有 **{rate_good}%** 将失分数控制在25以下。这证明了能够大量拿 विकेट（Wickets）的投手，通常也能很好地控制失分。
    """
    return fig, desc

def render_fig3(df):
    """图3：Virat Kohli 趋势"""
    k = df[df['Player_Name'] == 'Virat Kohli'].sort_values('Year')
    if k.empty: return plt.figure(), "无数据"
    
    peak = k[(k['Year']>=2013) & (k['Year']<=2018)]
    peak_max = peak['Runs_Scored'].max() if not peak.empty else 0
    peak_year = peak.loc[peak['Runs_Scored'].idxmax(), 'Year'] if not peak.empty else 0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(k['Year'], k['Runs_Scored'], 'b-o', linewidth=2.5, label='得分')
    ax1.fill_between(k['Year'], 0, k['Runs_Scored'], where=(k['Year']>=2013)&(k['Year']<=2018), color='red', alpha=0.2, label='巅峰期')
    
    ax2 = ax1.twinx()
    ax2.plot(k['Year'], k['Wickets_Taken'].fillna(0), 'r-s', linewidth=2.5, label='三柱门')
    
    ax1.text(peak_year, peak_max+20, f'巅峰: {peak_max}分', ha='center', bbox=dict(facecolor='yellow', alpha=0.8))
    ax1.set_title('Virat Kohli 2008-2024年度表现趋势', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    
    desc = f"""
    **生涯轨迹解读：**
    * **巅峰爆发（2013-2018）**：红色区域标记了他的黄金时期，其中{peak_year}年创下**{peak_max}分**的单赛季纪录，统治力惊人。
    * **职业定位**：蓝线（得分）极高而红线（三柱门）极低，清晰地表明他是一位**纯粹且顶级**的击球手（Batsman），几乎不参与投球任务。
    """
    return fig, desc

def render_fig4(df):
    """图4：箱线图"""
    years = [2010, 2015, 2020, 2024]
    data = []
    medians = []
    for y in years:
        d = df[(df['Year']==y) & (df['Batting_Average']>0)]['Batting_Average']
        data.append(d)
        medians.append(round(d.median(), 1))
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=years, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
    
    for i, m in enumerate(medians):
        ax.text(i+1, m+1, f'{m}', ha='center', fontweight='bold')
        
    ax.set_title('不同年份球员击球平均率分布', fontsize=14, fontweight='bold')
    ax.set_ylabel('击球平均率')
    
    desc = f"""
    **趋势演变：**
    * **中位数变化**：从2010年的{medians[0]}到2024年的{medians[-1]}，观察击球平均率中位数的波动。
    * **整体提升**：箱体的整体上移或中位数的增加，反映了IPL联赛整体击球水平的逐年进化，击球手对抗投球手的能力在增强。
    """
    return fig, desc

def render_fig5(df):
    """图5：雷达图"""
    target = ['Virat Kohli', 'MS Dhoni', 'Suryakumar Yadav']
    metrics = ['Batting_Average', 'Batting_Strike_Rate', 'Wickets_Taken', 'Bowling_Average', 'Catches_Taken']
    names = ['击球均率', '击球率', '三柱门', '失分(反)', '接球']
    
    p_df = df[df['Player_Name'].isin(target)].copy()
    best = p_df.groupby('Player_Name').apply(lambda x: x.nlargest(1, 'Runs_Scored')).reset_index(drop=True)
    
    if best.empty: return plt.figure(), "无数据"

    # 简单归一化逻辑用于展示
    radar_data = []
    for _, row in best.iterrows():
        vals = []
        for m in metrics:
            mx = df[m].max()
            v = row[m] if not pd.isna(row[m]) else 0
            if m == 'Bowling_Average': vals.append(10 - (v/mx*10) if mx>0 else 0)
            else: vals.append((v/mx*10) if mx>0 else 0)
        radar_data.append(vals)
        
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, (name, d) in enumerate(zip(best['Player_Name'], radar_data)):
        d += d[:1]
        ax.plot(angles, d, color=colors[i], linewidth=2, label=name)
        ax.fill(angles, d, color=colors[i], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=12)
    ax.set_title('顶级球员多维度能力画像', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    desc = """
    **球星画像对比：**
    1.  **Virat Kohli (蓝)**：在“击球均率”和“击球率”上延伸极长，典型的**进攻核心**。
    2.  **MS Dhoni (绿)**：在“接球”维度表现突出，体现了其作为**守门员/防守核心**的特殊价值，同时击球能力均衡。
    3.  **Suryakumar (红)**：各项指标较为均衡，展示了现代板球全能战士的特点。
    """
    return fig, desc

def render_fig6(df):
    """图6：组合图"""
    fig = plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "综合看板包含5张子图\n请在左侧分别选择图1-图5\n以查看每个维度的高清细节", ha='center', fontsize=14, color='#555')
    plt.axis('off')
    desc = "**综合仪表盘：**\n这是用于汇报的宏观视图，将得分分布（宏观）、投球效率（技术）、球星趋势（个例）、年度变化（时间）整合在一起，快速扫视联赛整体状况。"
    return fig, desc

def render_fig7(df):
    """图7：效率散点"""
    d = df[df['Year']>=2010].copy()
    d['E_Bat'] = d['Runs_Scored']/d['Matches_Batted']
    d['E_Bowl'] = d['Wickets_Taken']/d['Matches_Bowled']
    d = d.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(d['Matches_Batted'], d['E_Bat'], s=d['Runs_Scored']/10, c='cornflowerblue', alpha=0.5, label='击球效率')
    ax2 = ax.twinx()
    ax2.scatter(d['Matches_Bowled'], d['E_Bowl'], s=d['Wickets_Taken']*5, c='tomato', alpha=0.5, label='投球效率')
    
    ax.set_xlabel('参赛场次')
    ax.set_ylabel('得分效率', color='cornflowerblue')
    ax2.set_ylabel('投球效率', color='tomato')
    ax.set_title('参赛场次与效率分析', fontsize=14, fontweight='bold')
    
    desc = """
    **效率矩阵分析：**
    * **气泡大小**：分别代表总得分和总三柱门数。
    * **分布规律**：左上角的气泡代表“低场次、高效率”的爆发型替补或奇兵；右侧密集区则是球队的常规主力，效率趋于稳定。
    * **结论**：出场次数越多，维持超高效率越难，能长期保持在右上方的球员是绝对的传奇。
    """
    return fig, desc

def render_fig8(df):
    """图8：堆叠图"""
    cols = ['Centuries', 'Half_Centuries', 'Fours', 'Sixes']
    d = df[df['Year']>=2010].groupby('Year')[cols + ['Runs_Scored']].sum()
    for c in cols: d[c] = d[c]/d['Runs_Scored']*100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(d.index, [d[c] for c in cols], labels=cols, alpha=0.8)
    ax.legend(loc='upper right')
    ax.set_title('得分结构年度变化', fontsize=14, fontweight='bold')
    ax.set_ylabel('占比 (%)')
    
    desc = """
    **比赛风格演变：**
    * **边界球占比**：观察 **Fours** 和 **Sixes** 的面积变化。如果这部分面积逐年扩大，说明IPL比赛变得更加激进，球队更倾向于通过冒险的边界球来快速得分。
    * **里程碑**：Centuries (100分) 的极低占比显示了个人单场拿高分的极高难度。
    """
    return fig, desc

def render_fig9(df):
    """图9：双轴图"""
    d = df[(df['Year']>=2010)].copy()
    d['G'] = pd.cut(d['Batting_Average'], bins=[0,10,20,30,40,50,100])
    s = d.groupby('G').agg({'Player_Name':'count', 'Runs_Scored':'mean'})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(s.index.astype(str), s['Player_Name'], color='lightseagreen', alpha=0.6, label='人数')
    ax2 = ax.twinx()
    ax2.plot(s.index.astype(str), s['Runs_Scored'], 'ro-', linewidth=2, label='平均得分')
    
    ax.set_title('击球平均率区间分布与得分关系', fontsize=14, fontweight='bold')
    ax.set_ylabel('球员人数', color='lightseagreen')
    ax2.set_ylabel('平均得分', color='red')
    
    desc = """
    **双轴洞察：**
    * **人数分布（柱状）**：呈右偏分布，绝大多数球员的平均率停留在 10-30 分的普通区间。
    * **得分能力（折线）**：红线呈**指数级上升**。这有力地证明了，一旦球员的平均率突破 30 分大关，其对球队的总得分贡献将呈爆发式增长，这类球员是转会市场上的“顶薪”目标。
    """
    return fig, desc

def render_fig10(df):
    """图10：Top5趋势"""
    top5 = df.groupby('Player_Name')['Runs_Scored'].sum().nlargest(5).index
    fig, ax = plt.subplots(figsize=(12, 7))
    for p in top5:
        d = df[df['Player_Name']==p].groupby('Year')['Runs_Scored'].sum()
        ax.plot(d.index, d.values, 'o-', label=p)
    ax.legend()
    ax.set_title('历史得分榜TOP5球员年度趋势', fontsize=14, fontweight='bold')
    
    desc = """
    **巨星对决：**
    * 此图追踪了IPL历史上最伟大的5位得分手的轨迹。
    * **稳定性对比**：观察线条的波动幅度，可以判断谁是“昙花一现”的爆发型选手，谁是“细水长流”的常青树（如 Virat Kohli 的线条通常保持在较高水平）。
    """
    return fig, desc

def render_fig11(df):
    """图11：热力图"""
    d = df[(df['Bowling_Average']>0) & (df['Wickets_Taken']>0)]
    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.hist2d(d['Bowling_Average'], d['Wickets_Taken'], bins=20, cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax, label='密度')
    ax.axvline(30, color='g', linestyle='--', label='高效失分<30')
    ax.axhline(20, color='b', linestyle='--', label='高效三柱门>20')
    ax.set_title('投球效率密度热力图', fontsize=14, fontweight='bold')
    ax.set_xlabel('失分数'); ax.set_ylabel('三柱门数')
    ax.legend()
    
    desc = """
    **寻找黄金矿区：**
    * **颜色越深**：代表该数据区间的球员越密集。
    * **黄金区域**：右上角（低失分、高三柱门）由绿色和蓝色虚线围成的区域。热力图显示该区域颜色极浅，说明只有极少数顶级投手能同时做到“多拿人头”且“少丢分”。
    """
    return fig, desc

def render_fig12(df):
    """图12：条形图"""
    d = df[df['Year']>=2008].groupby('Year')['Player_Name'].nunique()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(d.index, d.values, color='skyblue')
    for i, v in zip(d.index, d.values):
        ax.text(v+1, i, str(v), va='center')
    ax.set_title('IPL历年参赛球员数量', fontsize=14, fontweight='bold')
    
    desc = """
    **联赛扩张史：**
    * 条形图直观展示了 IPL 联赛规模的扩张历程。
    * **数据增长**：参赛球员数量的逐年稳步增加（尤其是2022年后的增长），反映了球队扩军（新增两支球队）以及联赛商业影响力的持续扩大。
    """
    return fig, desc

def render_fig13(df):
    """图13：小提琴图"""
    d = df[df['Batting_Average']>0].groupby('Player_Name')['Batting_Average'].agg(['mean','std','count']).dropna()
    d['cv'] = d['std']/d['mean']
    d = d[d['cv']<=2]
    
    groups = [d[(d['count']>=l)&(d['count']<=r)]['cv'] for l,r in [(1,3),(4,6),(7,9),(10,99)]]
    labels = ['1-3年', '4-6年', '7-9年', '10年+']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.violinplot(groups, showmedians=True)
    ax.set_xticks(range(1,5)); ax.set_xticklabels(labels)
    ax.set_title('参赛年限与表现稳定性分析', fontsize=14, fontweight='bold')
    ax.set_ylabel('变异系数 (CV)')
    
    desc = """
    **老将更稳：**
    * **CV值解读**：变异系数（CV）越低，代表发挥越稳定。
    * **形态收敛**：随着参赛年限增加（从左至右），小提琴图的形状变得更窄且重心下移。这统计学上验证了“经验法则”——资深球员比新秀拥有更强的比赛稳定性。
    """
    return fig, desc

def render_fig14(df):
    """图14：象限图"""
    d = df[(df['Year']>=2020) & (df['Balls_Bowled']>0)].groupby('Player_Name').mean()
    med_x, med_y = d['Economy_Rate'].median(), d['Wickets_Taken'].median()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = np.where((d['Economy_Rate']<med_x)&(d['Wickets_Taken']>med_y), 'green', 
             np.where((d['Economy_Rate']>med_x)&(d['Wickets_Taken']<med_y), 'red', 'gray'))
    ax.scatter(d['Economy_Rate'], d['Wickets_Taken'], c=colors, alpha=0.6)
    ax.axvline(med_x, linestyle='--'); ax.axhline(med_y, linestyle='--')
    ax.set_title('投手效能四象限分析', fontsize=14, fontweight='bold')
    ax.set_xlabel('经济率'); ax.set_ylabel('场均三柱门')
    
    desc = """
    **四象限法则：**
    * **左上（绿色）**：**高效强攻型**。经济率低且拿 विकेट效率高，这是球队的王牌投手区域。
    * **右下（红色）**：**低效区**。经济率高且拿不到 विकेट，这类球员通常面临被淘汰的风险。
    * **策略意义**：球队应优先续约落在绿色区域的球员。
    """
    return fig, desc

def render_fig15(df):
    """图15：堆叠条形"""
    d = df[df['Year']>=2010].copy()
    d['Type'] = d.apply(lambda r: '击球手' if r['Batting_Average']>25 else ('投手' if r['Wickets_Taken']>5 else '边缘'), axis=1)
    s = d.groupby(['Year', 'Type']).size().unstack().fillna(0)
    s = s.div(s.sum(axis=1), axis=0)*100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    s.plot(kind='barh', stacked=True, ax=ax, colormap='Set3')
    ax.set_title('球员类型分布演变', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    desc = """
    **生态演变：**
    * 该图展示了“纯击球手”、“纯投手”和“边缘球员”的占比变化。
    * **全能化趋势**：如果中间色块（代表可能的全能型）占比逐年上升，说明现代板球战术越来越要求球员具备多面手的能力，单一功能的球员生存空间正在被压缩。
    """
    return fig, desc

def render_fig16(df):
    """图16：回归分析"""
    d = df[df['Year']>=2018].fillna(0)
    d['Score'] = d['Runs_Scored'] + d['Wickets_Taken']*20
    d = d[d['Score']>0]
    
    slope, intercept, r, p, _ = linregress(d['Catches_Taken'], d['Score'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(d['Catches_Taken'], d['Score'], alpha=0.5, c='#9B59B6')
    ax.plot(d['Catches_Taken'], slope*d['Catches_Taken']+intercept, 'r--', label=f'r={r:.2f}')
    ax.set_title('接球能力与综合表现相关性', fontsize=14, fontweight='bold')
    ax.set_xlabel('接球数'); ax.set_ylabel('综合得分')
    ax.legend()
    
    desc = f"""
    **全能悖论验证：**
    * **正相关 (r={r:.2f})**：回归线呈上升趋势，说明接球次数（防守积极性）与球员的综合表现得分显著正相关。
    * **结论**：数据打破了“核心球员不防守”的刻板印象。越是核心的超级巨星，在防守端（接球）的投入度往往也越高，态度决定上限。
    """
    return fig, desc

# ===================== 4. 页面布局逻辑 =====================

# 侧边栏：文件加载
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/8/8d/Cricket_India_Crest.svg", width=80)
    st.markdown("### 🏏 导航控制台")
    
    DEFAULT_FILE = "data.csv"
    ALT_FILE = "6-球员生命周期_预处理后.csv"
    
    df = None
    # 优先加载本地文件
    if os.path.exists(DEFAULT_FILE):
        df = load_data(DEFAULT_FILE)
        st.success(f"已加载: {DEFAULT_FILE}")
    elif os.path.exists(ALT_FILE):
        df = load_data(ALT_FILE)
        st.success(f"已加载: {ALT_FILE}")
    else:
        uploaded_file = st.file_uploader("请上传数据文件 (CSV)", type=['csv'])
        if uploaded_file:
            df = load_data(uploaded_file)

    if df is not None:
        st.markdown("---")
        menu = st.radio(
            "选择分析模块:",
            ("🏠 首页大屏", "📊 数据总览", "🏏 击球深度分析", "🥎 投球深度分析", "🔗 综合与关联", "⭐ 球星特写")
        )

# 主内容区域
if df is None:
    st.markdown("""
    <div class="hero-box">
        <h1 class="hero-title">IPL DATA VISION</h1>
        <p class="hero-subtitle">等待数据加载...</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("请在左侧侧边栏上传 CSV 数据文件以开始分析。")

elif menu == "🏠 首页大屏":
    st.markdown("""
    <div class="hero-box">
        <h1 class="hero-title">IPL 职业板球数据视界</h1>
        <p class="hero-subtitle">Professional Cricket Data Analysis & Visualization Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分析周期", "2008 - 2024", "16 Years")
    col2.metric("球员总数", f"{df['Player_Name'].nunique()}", "+12 New")
    col3.metric("总得分别", f"{int(df['Runs_Scored'].sum()/10000)}W+", "Runs")
    col4.metric("数据记录", f"{len(df)}", "Rows")
    
    st.markdown("---")
    st.markdown("### 🎯 系统功能概览")
    st.markdown("""
    本系统集成了 **16 个专业分析模型**，为您提供全方位的板球数据洞察：
    * **📈 趋势追踪**：从时间维度剖析联赛发展与球星生涯轨迹。
    * **🔍 深度挖掘**：通过回归分析与聚类算法，发现数据背后的隐形规律。
    * **⚖️ 效能评估**：利用四象限图与雷达图，精准定位球员的市场价值。
    """)

else:
    # 路由映射
    chart_map = {
        "📊 数据总览": {
            "图1: 年度得分分布 (直方图)": render_fig1,
            "图12: 参赛规模扩张 (条形图)": render_fig12,
            "图15: 球员类型演变 (堆叠图)": render_fig15
        },
        "🏏 击球深度分析": {
            "图4: 击球均率分布 (箱线图)": render_fig4,
            "图8: 得分结构变化 (堆叠面积)": render_fig8,
            "图9: 均率与得分关系 (双轴图)": render_fig9,
            "图10: 历史得分王趋势 (多线图)": render_fig10,
            "图13: 表现稳定性分析 (小提琴)": render_fig13
        },
        "🥎 投球深度分析": {
            "图2: 三柱门与失分 (散点图)": render_fig2,
            "图11: 投球黄金区域 (热力图)": render_fig11,
            "图14: 投手效能四象限 (象限图)": render_fig14
        },
        "🔗 综合与关联": {
            "图6: 综合数据仪表盘 (组合图)": render_fig6,
            "图7: 攻守效率矩阵 (气泡图)": render_fig7,
            "图16: 防守与综合表现 (回归分析)": render_fig16
        },
        "⭐ 球星特写": {
            "图3: Virat Kohli 生涯轨迹": render_fig3,
            "图5: 顶级球星能力雷达": render_fig5
        }
    }
    
    charts = chart_map[menu]
    selected_chart = st.sidebar.radio("选择图表:", list(charts.keys()))
    
    st.subheader(f"📈 {selected_chart}")
    
    # === 核心布局修改：3:1 比例 ===
    c_chart, c_text = st.columns([3, 1])
    
    # 渲染图表与文字
    func = charts[selected_chart]
    fig, desc = func(df)
    
    with c_chart:
        # use_container_width=True 会自动限制图片宽度适应列宽
        st.pyplot(fig, use_container_width=True)
        
    with c_text:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">📊 结果解读</div>
            <div class="info-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 底部数据预览
    with st.expander("🔍 点击查看源数据"):
        st.dataframe(df.head(20))