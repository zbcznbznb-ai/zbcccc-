import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy import stats
import matplotlib.font_manager as fm
import os
import warnings

# ===================== 1. 基础配置 =====================
st.set_page_config(
    page_title="IPL 球员生命周期可视化分析系统 (原作复刻版)",
    page_icon="🏏",
    layout="wide"
)

warnings.filterwarnings('ignore')

# ----------------- 字体智能加载 (保留之前的优化) -----------------
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

# ===================== 2. 数据预处理 (复刻“数据集预处理.py”) =====================
@st.cache_data
def load_and_process_data(file):
    # 读取数据
    df = pd.read_csv(file)
    
    # 1. 关键列处理 (策略1)
    if 'Player_Name' in df.columns and 'Year' in df.columns:
        df = df.dropna(subset=['Player_Name', 'Year'])

    # 2. 异常值标记替换 (策略2)
    stats_columns = ['Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
                    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries', 'Fours', 'Sixes',
                    'Catches_Taken', 'Stumpings', 'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded',
                    'Wickets_Taken', 'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate',
                    'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls']
    
    for col in stats_columns:
        if col in df.columns:
            df[col] = df[col].replace('No stats', np.nan)
            # 策略3: 转换为数值
            if col not in ['Best_Bowling_Match', 'Highest_Score']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 一致性检测
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Batting_Average' in df.columns:
        df.loc[df['Batting_Average'] > 100, 'Batting_Average'] = np.nan
    if 'Bowling_Average' in df.columns:
        df.loc[df['Bowling_Average'] > 100, 'Bowling_Average'] = np.nan
    if 'Player_Name' in df.columns:
        df['Player_Name'] = df['Player_Name'].str.strip()
        
    # 2. 重复值处理
    df['核心键'] = df['Player_Name'].astype(str) + '_' + df['Year'].astype(str).fillna('NaN')
    df = df.drop_duplicates(subset=['核心键'], keep='first')
    df.drop('核心键', axis=1, inplace=True)

    return df

# ===================== 3. 绘图函数集 (1:1 复刻原作) =====================

def plot_fig1(df):
    """图1：球员年度总跑位得分分布直方图"""
    # 复刻原代码逻辑
    valid_runs = df[df['Runs_Scored'].notna()].copy()
    valid_runs['Runs_Scored'] = pd.to_numeric(valid_runs['Runs_Scored'], errors='coerce')
    valid_runs = valid_runs[valid_runs['Runs_Scored'] > 0]

    range_0_150 = len(valid_runs[(valid_runs['Runs_Scored'] >= 0) & (valid_runs['Runs_Scored'] <= 150)])
    range_500_plus = len(valid_runs[valid_runs['Runs_Scored'] >= 500])
    rate_0_150 = round((range_0_150 / len(valid_runs) * 100), 1)
    rate_500_plus = round((range_500_plus / len(valid_runs) * 100), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(valid_runs['Runs_Scored'], bins=30, color='steelblue', 
                               edgecolor='black', alpha=0.8)

    for i, patch in enumerate(patches):
        if bins[i] >= 0 and bins[i+1] <= 150:
            patch.set_facecolor('orange')

    ax.text(75, max(n)*0.8, f'0-150分区间占比：{rate_0_150}%', 
            ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(700, max(n)*0.5, f'500+分区间占比：{rate_500_plus}%', 
            ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax.set_title('球员年度总跑位得分分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('总跑位得分', fontsize=12)
    ax.set_ylabel('球员人数', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_fig2(df):
    """图2：三柱门数与投球平均失分数散点图"""
    valid_bowling = df[(df['Wickets_Taken'].notna()) & (df['Bowling_Average'].notna())].copy()
    valid_bowling['Wickets_Taken'] = pd.to_numeric(valid_bowling['Wickets_Taken'], errors='coerce')
    valid_bowling['Bowling_Average'] = pd.to_numeric(valid_bowling['Bowling_Average'], errors='coerce')
    valid_bowling = valid_bowling[(valid_bowling['Wickets_Taken'] > 0) & (valid_bowling['Bowling_Average'] > 0)]

    corr, _ = pearsonr(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'])
    corr_rounded = round(corr, 2)

    wickets_gt15 = valid_bowling[valid_bowling['Wickets_Taken'] > 15]
    rate_gt15_below25 = 0
    if len(wickets_gt15) > 0:
        rate_gt15_below25 = round((len(wickets_gt15[wickets_gt15['Bowling_Average'] < 25]) / len(wickets_gt15) * 100), 1)

    wickets_lt5 = valid_bowling[valid_bowling['Wickets_Taken'] < 5]
    rate_lt5_above30 = 0
    if len(wickets_lt5) > 0:
        rate_lt5_above30 = round((len(wickets_lt5[wickets_lt5['Bowling_Average'] > 30]) / len(wickets_lt5) * 100), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid_bowling['Wickets_Taken'], valid_bowling['Bowling_Average'], 
               alpha=0.6, color='coral', s=40, edgecolor='white', linewidth=0.5)
    ax.scatter(wickets_gt15['Wickets_Taken'], wickets_gt15['Bowling_Average'], 
               color='darkgreen', s=60, alpha=0.8, label=f'三柱门数>15（{rate_gt15_below25}%失分数<25）')
    ax.scatter(wickets_lt5['Wickets_Taken'], wickets_lt5['Bowling_Average'], 
               color='darkred', s=60, alpha=0.8, label=f'三柱门数<5（{rate_lt5_above30}%失分数>30）')

    ax.text(valid_bowling['Wickets_Taken'].max()*0.7, valid_bowling['Bowling_Average'].max()*0.8, 
            f'Pearson相关系数：{corr_rounded}', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_title('三柱门数与投球平均失分数关系', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('三柱门数', fontsize=12)
    ax.set_ylabel('投球平均失分数', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    return fig

def plot_fig3(df):
    """图3：Virat Kohli 年度表现趋势线图"""
    kohli_df = df[df['Player_Name'] == 'Virat Kohli'].copy()
    if kohli_df.empty: return plt.figure()
    
    kohli_df = kohli_df[(kohli_df['Year'].notna()) & (kohli_df['Runs_Scored'].notna())]
    kohli_df['Year'] = pd.to_numeric(kohli_df['Year'], errors='coerce').astype(int)
    kohli_df['Runs_Scored'] = pd.to_numeric(kohli_df['Runs_Scored'], errors='coerce')
    kohli_df['Wickets_Taken'] = pd.to_numeric(kohli_df['Wickets_Taken'], errors='coerce').fillna(0)
    kohli_df = kohli_df.sort_values('Year')

    growth_phase = kohli_df[(kohli_df['Year'] >= 2008) & (kohli_df['Year'] <= 2012)]
    peak_phase = kohli_df[(kohli_df['Year'] >= 2013) & (kohli_df['Year'] <= 2018)]
    stable_phase = kohli_df[(kohli_df['Year'] >= 2019) & (kohli_df['Year'] <= 2024)]
    
    peak_max_score = peak_phase['Runs_Scored'].max() if not peak_phase.empty else 0
    peak_year = peak_phase[peak_phase['Runs_Scored'] == peak_max_score]['Year'].iloc[0] if not peak_phase.empty else 2015

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(kohli_df['Year'], kohli_df['Runs_Scored'], 'b-o', linewidth=2.5, markersize=6, label='总跑位得分')
    ax1.fill_between(growth_phase['Year'], 0, growth_phase['Runs_Scored'], alpha=0.2, color='blue', label='成长期（2008-2012）')
    ax1.fill_between(peak_phase['Year'], 0, peak_phase['Runs_Scored'], alpha=0.2, color='red', label='巅峰期（2013-2018）')
    ax1.fill_between(stable_phase['Year'], 0, stable_phase['Runs_Scored'], alpha=0.2, color='green', label='稳定期（2019-2024）')

    ax2 = ax1.twinx()
    ax2.plot(kohli_df['Year'], kohli_df['Wickets_Taken'], 'r-s', linewidth=2.5, markersize=6, label='三柱门数')

    ax1.text(peak_year, peak_max_score + 20, f'巅峰期最高：{peak_max_score}分\n（{peak_year}年）', 
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_title('Virat Kohli 2008-2024年度表现趋势', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('年份', fontsize=12)
    ax1.set_ylabel('总跑位得分', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel('三柱门数', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    return fig

def plot_fig4(df):
    """图4：不同年份球员击球平均率箱线图"""
    target_years = [2010, 2015, 2020, 2024]
    valid_batting = df[(df['Year'].notna()) & (df['Batting_Average'].notna())].copy()
    valid_batting['Year'] = pd.to_numeric(valid_batting['Year'], errors='coerce').astype(int)
    valid_batting['Batting_Average'] = pd.to_numeric(valid_batting['Batting_Average'], errors='coerce')
    valid_batting = valid_batting[(valid_batting['Year'].isin(target_years)) & (valid_batting['Batting_Average'] > 0)]

    yearly_stats = {}
    for year in target_years:
        year_data = valid_batting[valid_batting['Year'] == year]['Batting_Average']
        if len(year_data) > 5:
            median = round(year_data.median(), 1)
            q1 = year_data.quantile(0.25)
            q3 = year_data.quantile(0.75)
            iqr = round((q3 - q1), 1)
            yearly_stats[year] = {'median': median, 'iqr': iqr, 'data': year_data}

    valid_years = list(yearly_stats.keys())
    yearly_data = [yearly_stats[year]['data'] for year in valid_years]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(yearly_data, labels=valid_years, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', alpha=0.8),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))

    for i, year in enumerate(valid_years):
        median = yearly_stats[year]['median']
        iqr = yearly_stats[year]['iqr']
        ax.text(i+1, median + 1, f'中位数：{median}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i+1, yearly_stats[year]['data'].min() - 5, f'IQR：{iqr}', ha='center', fontsize=9, fontweight='bold')

    ax.set_title('不同年份球员击球平均率分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('击球平均率', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_fig5(df):
    """图5：顶级球员多维度雷达图"""
    target_players = ['Virat Kohli', 'MS Dhoni', 'Suryakumar Yadav']
    indicators = ['击球平均率', '击球率', '三柱门数', '投球平均失分数（反向）', '接球次数']
    col_mapping = {
        '击球平均率': 'Batting_Average', '击球率': 'Batting_Strike_Rate',
        '三柱门数': 'Wickets_Taken', '投球平均失分数（反向）': 'Bowling_Average',
        '接球次数': 'Catches_Taken'
    }

    player_df = df[df['Player_Name'].isin(target_players)].copy()
    for dim in indicators:
        col = col_mapping[dim]
        if col in player_df.columns:
            player_df[col] = pd.to_numeric(player_df[col], errors='coerce').fillna(0)

    best_year_data = []
    for player in target_players:
        p_data = player_df[player_df['Player_Name'] == player].copy()
        if len(p_data) > 0:
            p_data = p_data.sort_values(by=['Runs_Scored', 'Year'], ascending=[False, False])
            best_year_data.append(p_data.iloc[0])
    best_df = pd.DataFrame(best_year_data)
    if best_df.empty: return plt.figure()

    def normalize_indicator(value, min_val, max_val, is_reverse=False):
        if max_val == min_val: return 5.0
        if is_reverse:
            norm_score = 10 - ((value - min_val) / (max_val - min_val)) * 10
        else:
            norm_score = ((value - min_val) / (max_val - min_val)) * 10
        return max(0.0, min(10.0, norm_score))

    extremes = {}
    for dim in indicators:
        col = col_mapping[dim]
        valid_vals = best_df[best_df[col] > 0][col]
        extremes[dim] = (valid_vals.min(), valid_vals.max()) if len(valid_vals) > 0 else (0, 1)

    radar_data = []
    for _, row in best_df.iterrows():
        player_scores = []
        for dim in indicators:
            col = col_mapping[dim]
            min_val, max_val = extremes[dim]
            is_reverse = dim == '投球平均失分数（反向）'
            player_scores.append(round(normalize_indicator(row[col], min_val, max_val, is_reverse), 1))
        radar_data.append(player_scores)

    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]
    radar_data_closed = [scores + scores[:1] for scores in radar_data]
    indicators_closed = indicators + indicators[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    markers = ['o', 's', '^']

    for i, (player, scores, color, marker) in enumerate(zip(target_players, radar_data_closed, colors, markers)):
        ax.plot(angles, scores, color=color, linewidth=2.5, marker=marker, markersize=8, label=player)
        ax.fill(angles, scores, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators_closed[:-1], fontsize=11, fontweight='bold')
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_title('IPL顶级球员多维度表现对比雷达图\n', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    return fig

def plot_fig7(df):
    """图7：效率散点图"""
    core_cols = ['Year', 'Matches_Batted', 'Runs_Scored', 'Batting_Average', 'Matches_Bowled', 'Wickets_Taken', 'Bowling_Average']
    df_valid = df.dropna(subset=core_cols).copy()
    df_valid = df_valid[(df_valid['Year'] >= 2010) & (df_valid['Year'] <= 2024)]

    df_valid['得分效率'] = df_valid['Runs_Scored'] / df_valid['Matches_Batted']
    df_valid['投球效率'] = df_valid['Wickets_Taken'] / df_valid['Matches_Bowled']
    df_valid['得分效率'] = df_valid['得分效率'].replace([np.inf, -np.inf], 0).fillna(0)
    df_valid['投球效率'] = df_valid['投球效率'].replace([np.inf, -np.inf], 0).fillna(0)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.scatter(df_valid['Matches_Batted'], df_valid['得分效率'], s=df_valid['Runs_Scored']/10,
                c='cornflowerblue', alpha=0.6, edgecolors='white', linewidth=0.5, label='击球得分效率')
    ax1.set_xlabel('击球参赛场次', fontsize=12)
    ax1.set_ylabel('得分效率（每场次得分）', fontsize=12, color='cornflowerblue')
    ax1.tick_params(axis='y', labelcolor='cornflowerblue')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.scatter(df_valid['Matches_Bowled'], df_valid['投球效率'], s=df_valid['Wickets_Taken']*5,
                c='tomato', alpha=0.6, edgecolors='white', linewidth=0.5, label='投球三柱门效率')
    ax2.set_ylabel('投球效率（每场次三柱门数）', fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')

    top_bat = df_valid[df_valid['得分效率'] > 50].iloc[0] if len(df_valid[df_valid['得分效率'] > 50]) > 0 else None
    if top_bat is not None:
        ax1.annotate(f'{top_bat["Player_Name"]}\n得分效率：{top_bat["得分效率"]:.1f}',
                     xy=(top_bat['Matches_Batted'], top_bat['得分效率']), xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=9)
    
    top_bowl = df_valid[df_valid['投球效率'] > 2].iloc[0] if len(df_valid[df_valid['投球效率'] > 2]) > 0 else None
    if top_bowl is not None:
        ax2.annotate(f'{top_bowl["Player_Name"]}\n投球效率：{top_bowl["投球效率"]:.1f}',
                     xy=(top_bowl['Matches_Bowled'], top_bowl['投球效率']), xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

    plt.title('球员参赛场次与得分/投球效率关系分析（2010-2024）', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    return fig

def plot_fig8(df):
    """图8：得分结构堆叠图"""
    core_cols = ['Year', 'Centuries', 'Half_Centuries', 'Fours', 'Sixes', 'Runs_Scored']
    df_valid = df.dropna(subset=core_cols).copy()
    df_valid = df_valid[(df_valid['Year'] >= 2010) & (df_valid['Year'] <= 2024)]
    score_cols = ['Centuries', 'Half_Centuries', 'Fours', 'Sixes']
    yearly_score = df_valid.groupby('Year')[score_cols + ['Runs_Scored']].sum()
    
    for col in score_cols:
        yearly_score[col + '_占比'] = np.where(yearly_score['Runs_Scored'] > 0, yearly_score[col] / yearly_score['Runs_Scored'] * 100, 0)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax.stackplot(yearly_score.index, [yearly_score[col + '_占比'] for col in score_cols],
                 labels=[col for col in score_cols], colors=colors, alpha=0.8)

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('得分结构占比（%）', fontsize=12)
    ax.set_title('2010-2024年球员得分结构年度变化（堆叠面积图）', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    return fig

def plot_fig9(df):
    """图9：平均率区间分布"""
    core_cols = ['Year', 'Batting_Average', 'Runs_Scored', 'Player_Name']
    df_valid = df.dropna(subset=core_cols).copy()
    df_valid = df_valid[(df_valid['Year'] >= 2010) & (df_valid['Year'] <= 2024)]
    bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '50+']
    df_valid['平均率区间'] = pd.cut(df_valid['Batting_Average'], bins=bins, labels=labels, right=True)
    
    interval_stats = df_valid.groupby('平均率区间').agg({'Player_Name': 'count', 'Runs_Scored': 'mean'}).reset_index()
    interval_stats.columns = ['平均率区间', '球员数量', '区间平均得分']

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bars = ax1.bar(interval_stats['平均率区间'], interval_stats['球员数量'],
                   color='lightseagreen', alpha=0.7, edgecolor='black', linewidth=0.5, label='球员数量')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('击球平均率区间', fontsize=12)
    ax1.set_ylabel('球员数量', fontsize=12, color='lightseagreen')
    ax1.tick_params(axis='y', labelcolor='lightseagreen')
    ax1.grid(alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    ax2.plot(interval_stats['平均率区间'], interval_stats['区间平均得分'], 'ro-', linewidth=2, markersize=6, label='区间平均得分')
    for x, y in zip(interval_stats['平均率区间'], interval_stats['区间平均得分']):
        ax2.text(x, y + 5, f'{int(y)}', ha='center', va='bottom', fontsize=10, color='red')
    
    ax2.set_ylabel('区间平均得分', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title('不同击球平均率区间球员数量与得分分布', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10)
    return fig

def plot_fig10(df):
    """图10：TOP5球员趋势"""
    core_cols = ['Year', 'Player_Name', 'Runs_Scored']
    df_valid = df.dropna(subset=core_cols).copy()
    df_valid = df_valid[(df_valid['Year'] >= 2010) & (df_valid['Year'] <= 2024)]
    top5_players = df_valid.groupby('Player_Name')['Runs_Scored'].sum().nlargest(5).index
    top5_data = df_valid[df_valid['Player_Name'].isin(top5_players)]
    player_yearly = top5_data.groupby(['Player_Name', 'Year'])['Runs_Scored'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    for i, player in enumerate(top5_players):
        player_data = player_yearly[player_yearly['Player_Name'] == player]
        ax.plot(player_data['Year'], player_data['Runs_Scored'], color=colors[i], linewidth=2.5, marker='o', markersize=6, label=player)
        if len(player_data) > 0:
            peak_year = player_data.loc[player_data['Runs_Scored'].idxmax(), 'Year']
            peak_score = player_data['Runs_Scored'].max()
            ax.annotate(f'峰值：{int(peak_score)}', xy=(peak_year, peak_score), xytext=(1, 10), textcoords='offset points', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', color=colors[i], alpha=0.5))

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('年度得分', fontsize=12)
    ax.set_title('生涯总得分TOP5球员年度得分趋势对比（2010-2024）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    return fig

def plot_fig11(df):
    """图11：投球效率热力图"""
    core_cols = ['Year', 'Bowling_Average', 'Wickets_Taken']
    df_valid = df.dropna(subset=core_cols).copy()
    df_valid = df_valid[(df_valid['Year'] >= 2010) & (df_valid['Year'] <= 2024)]
    bowl_data = df_valid[(df_valid['Bowling_Average'] > 0) & (df_valid['Wickets_Taken'] > 0)]

    fig, ax = plt.subplots(figsize=(12, 8))
    if len(bowl_data) > 0:
        hist, xedges, yedges = np.histogram2d(bowl_data['Bowling_Average'], bowl_data['Wickets_Taken'], bins=20, density=True)
        im = ax.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='YlOrRd', aspect='auto')
        cbar = plt.colorbar(im)
        cbar.set_label('密度（球员数量/区间）', fontsize=11)

    ax.axvline(x=30, color='green', linestyle='--', alpha=0.8, label='高效失分数阈值（<30）')
    ax.axhline(y=20, color='blue', linestyle='--', alpha=0.8, label='高效三柱门数阈值（>20）')
    ax.set_xlabel('投球平均失分数（越低越好）', fontsize=12)
    ax.set_ylabel('年度三柱门数（越高越好）', fontsize=12)
    ax.set_title('投球平均失分数与三柱门数密度分布热力图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    return fig

def plot_fig12(df):
    """图12：参赛年份分布"""
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    valid_years = df[df['Year'].notna() & (df['Year'] >= 2008) & (df['Year'] <= 2024)]
    yearly_players = valid_years.groupby('Year')['Player_Name'].nunique().reset_index()
    yearly_players = yearly_players[yearly_players['Player_Name'] >= 10].sort_values('Year')

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(yearly_players)))
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(yearly_players['Year'].astype(int), yearly_players['Player_Name'], color=colors, edgecolor='white', linewidth=1)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', ha='left', va='center', fontweight='bold', fontsize=10)

    ax.set_title('2008-2024年IPL联赛参赛球员数量分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('参赛球员数量', fontsize=12)
    ax.set_ylabel('年份', fontsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    return fig

def plot_fig13(df):
    """图13：稳定性分析 (小提琴图)"""
    df_valid = df[(df['Batting_Average'] > 0) & (df['Year'].notna()) & (df['Player_Name'].notna())].copy()
    player_career = df_valid.groupby('Player_Name').agg(
        首参赛年份=('Year', 'min'), 末参赛年份=('Year', 'max'), 击球平均率列表=('Batting_Average', list)
    ).reset_index()
    player_career['参赛年限'] = player_career['末参赛年份'] - player_career['首参赛年份'] + 1
    player_career['平均击球率'] = player_career['击球平均率列表'].apply(lambda x: np.mean(x))
    player_career['击球率标准差'] = player_career['击球平均率列表'].apply(lambda x: np.std(x))
    player_career['波动系数'] = player_career['击球率标准差'] / player_career['平均击球率']
    player_career = player_career[player_career['波动系数'] <= 2.0]

    def career_group(years):
        if years <= 3: return '1-3年（新秀期）'
        elif years <= 6: return '4-6年（成长期）'
        elif years <= 9: return '7-9年（巅峰期）'
        else: return '10年+（资深期）'
    player_career['参赛年限分组'] = player_career['参赛年限'].apply(career_group)

    fig, ax = plt.subplots(figsize=(12, 7))
    groups = ['1-3年（新秀期）', '4-6年（成长期）', '7-9年（巅峰期）', '10年+（资深期）']
    data = [player_career[player_career['参赛年限分组'] == g]['波动系数'].dropna() for g in groups]
    
    parts = ax.violinplot(data, positions=range(len(groups)), showmeans=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        parts[partname].set_edgecolor('black')
        parts[partname].set_linewidth(1.5)

    for i, g in enumerate(groups):
        group_data = player_career[player_career['参赛年限分组'] == g]['波动系数'].dropna()
        ax.text(i, max(group_data) + 0.05, f'n={len(group_data)}\n平均波动系数：{round(group_data.mean(), 2)}', ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='稳定阈值（波动系数=0.5）')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('击球平均率波动系数（越小越稳定）', fontsize=12)
    ax.set_title('球员参赛年限与表现稳定性分析', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)
    return fig

def plot_fig14(df):
    """图14：投手象限分析"""
    df_pitcher = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)].copy()
    numeric_cols = ['Economy_Rate', 'Wickets_Taken', 'Balls_Bowled', 'Matches_Bowled']
    for col in numeric_cols:
        df_pitcher[col] = pd.to_numeric(df_pitcher[col], errors='coerce').fillna(0)
    df_pitcher = df_pitcher[(df_pitcher['Balls_Bowled'] > 0) & (df_pitcher['Economy_Rate'] > 0)]

    pitcher_stats = df_pitcher.groupby('Player_Name').agg(
        平均经济率=('Economy_Rate', 'mean'), 总三柱门数=('Wickets_Taken', 'sum'),
        总投球数=('Balls_Bowled', 'sum'), 总投球场次=('Matches_Bowled', 'sum')
    ).reset_index()
    pitcher_stats['三柱门效率'] = (pitcher_stats['总三柱门数'] / pitcher_stats['总投球数']) * 100
    pitcher_stats = pitcher_stats[(pitcher_stats['平均经济率'] < 15) & (pitcher_stats['三柱门效率'] < 15)]

    eco_median = pitcher_stats['平均经济率'].median()
    wicket_median = pitcher_stats['三柱门效率'].median()

    def quadrant(row):
        if row['平均经济率'] < eco_median and row['三柱门效率'] > wicket_median: return '高效强攻型（Q1）', '#27AE60'
        elif row['平均经济率'] < eco_median and row['三柱门效率'] <= wicket_median: return '高效稳健型（Q2）', '#3498DB'
        elif row['平均经济率'] >= eco_median and row['三柱门效率'] > wicket_median: return '低效强攻型（Q3）', '#F39C12'
        else: return '低效稳健型（Q4）', '#E74C3C'

    pitcher_stats[['象限类型', '颜色']] = pitcher_stats.apply(lambda x: pd.Series(quadrant(x)), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    for quadrant, color in [('高效强攻型（Q1）', '#27AE60'), ('高效稳健型（Q2）', '#3498DB'), ('低效强攻型（Q3）', '#F39C12'), ('低效稳健型（Q4）', '#E74C3C')]:
        quad_data = pitcher_stats[pitcher_stats['象限类型'] == quadrant]
        ax.scatter(quad_data['平均经济率'], quad_data['三柱门效率'], s=quad_data['总投球场次']*2, c=color, alpha=0.6, edgecolors='white', linewidth=0.5, label=f'{quadrant}（n={len(quad_data)}）')

    ax.axvline(x=eco_median, color='black', linestyle='--', alpha=0.7, label=f'经济率中位数：{eco_median:.2f}')
    ax.axhline(y=wicket_median, color='black', linestyle='--', alpha=0.7, label=f'三柱门效率中位数：{wicket_median:.2f}')
    
    for quadrant in ['高效强攻型（Q1）', '高效稳健型（Q2）', '低效强攻型（Q3）', '低效稳健型（Q4）']:
        quad_data = pitcher_stats[pitcher_stats['象限类型'] == quadrant]
        if len(quad_data) > 0:
            top_player = quad_data.nlargest(1, '总投球场次').iloc[0]
            ax.annotate(f"{top_player['Player_Name']}\n场次：{int(top_player['总投球场次'])}", xy=(top_player['平均经济率'], top_player['三柱门效率']), xytext=(5, 5), textcoords='offset points', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=9)

    ax.set_xlabel('平均经济率（越低越好）', fontsize=12)
    ax.set_ylabel('三柱门效率（每100球三柱门数，越高越好）', fontsize=12)
    ax.set_title('2020-2024年投手经济率与三柱门效率象限分析', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    return fig

def plot_fig15(df):
    """图15：球员类型分布"""
    df_valid = df[(df['Year'] >= 2010) & (df['Year'] <= 2024) & (df['Player_Name'].notna())].copy()
    df_valid['Batting_Average'] = pd.to_numeric(df_valid['Batting_Average'], errors='coerce').fillna(0)
    df_valid['Wickets_Taken'] = pd.to_numeric(df_valid['Wickets_Taken'], errors='coerce').fillna(0)

    def player_type(row):
        ba, wt = row['Batting_Average'], row['Wickets_Taken']
        if ba >= 25 and wt <= 2: return '纯击球手'
        elif wt >= 5 and ba <= 15: return '纯投手'
        elif ba >= 20 and wt >= 3: return '全能型'
        else: return '边缘型'
    df_valid['球员类型'] = df_valid.apply(player_type, axis=1)

    yearly_type = df_valid.groupby(['Year', '球员类型']).size().unstack(fill_value=0)
    yearly_type['总球员数'] = yearly_type.sum(axis=1)
    for col in ['纯击球手', '纯投手', '全能型', '边缘型']:
        yearly_type[f'{col}占比'] = (yearly_type[col] / yearly_type['总球员数'] * 100).round(1)
    stack_data = yearly_type[['纯击球手占比', '纯投手占比', '全能型占比', '边缘型占比']].sort_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#45B7D1', '#FF6B6B', '#2ECC71', '#95A5A6']
    labels = ['纯击球手', '纯投手', '全能型', '边缘型']
    bottom = np.zeros(len(stack_data))
    for i, (col, color) in enumerate(zip(stack_data.columns, colors)):
        ax.barh(stack_data.index, stack_data[col], left=bottom, color=color, label=labels[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        for j, (idx, value) in enumerate(stack_data[col].items()):
            if value > 5: ax.text(bottom[j] + value/2, idx, f'{value}%', ha='center', va='center', fontweight='bold', fontsize=9)
        bottom += stack_data[col].values

    for idx, total in yearly_type['总球员数'].items():
        ax.text(102, idx, f'n={int(total)}', ha='left', va='center', fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_title('2010-2024年IPL联赛球员类型分布变化', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    return fig

def plot_fig16(df):
    """图16：接球与综合表现"""
    df_recent = df[(df['Year'] >= 2018) & (df['Year'] <= 2024)].copy()
    for col in ['Catches_Taken', 'Runs_Scored', 'Wickets_Taken', 'Matches_Batted', 'Matches_Bowled']:
        df_recent[col] = pd.to_numeric(df_recent[col], errors='coerce').fillna(0)

    def manual_normalize(value, min_val, max_val):
        return ((value - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 0

    min_runs, max_runs = df_recent['Runs_Scored'].min(), df_recent['Runs_Scored'].max()
    min_wickets, max_wickets = df_recent['Wickets_Taken'].min(), df_recent['Wickets_Taken'].max()
    df_recent['标准化击球'] = df_recent['Runs_Scored'].apply(lambda x: manual_normalize(x, min_runs, max_runs))
    df_recent['标准化三柱门'] = df_recent['Wickets_Taken'].apply(lambda x: manual_normalize(x, min_wickets, max_wickets))
    df_recent['综合表现得分'] = df_recent['标准化击球'] * 0.6 + df_recent['标准化三柱门'] * 0.4

    df_analysis = df_recent[(df_recent['综合表现得分'] > 0) & (df_recent['Catches_Taken'] >= 0)]
    x, y = df_analysis['Catches_Taken'], df_analysis['综合表现得分']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(x, y, s=df_analysis['Matches_Batted'] + df_analysis['Matches_Bowled'], c='#9B59B6', alpha=0.6, edgecolors='white', linewidth=0.5)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'回归线（r={r_value:.2f}）')

    avg_catches, avg_perf = x.mean(), y.mean()
    ax.axvline(x=avg_catches, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=avg_perf, color='black', linestyle='--', alpha=0.5)
    
    ax.text(avg_catches+2, avg_perf+10, '高接球+高表现\n（全能核心）', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(avg_catches-5, avg_perf+10, '低接球+高表现\n（进攻核心）', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(avg_catches+2, avg_perf-10, '高接球+低表现\n（防守 specialists）', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(avg_catches-5, avg_perf-10, '低接球+低表现\n（边缘球员）', ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax.set_title(f'球员接球能力与综合表现相关性分析（2018-2024）\nPearson相关系数：{r_value:.2f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('年度接球次数')
    ax.set_ylabel('综合表现得分')
    return fig

# ===================== 4. Streamlit 页面布局 =====================
st.title("🏏 IPL 顶级球员生命周期与表现可视化系统")
st.markdown("---")

DEFAULT_FILE = "data.csv"
ALT_FILE = "6-球员生命周期_预处理后.csv"

df = None
if os.path.exists(DEFAULT_FILE):
    df = load_and_process_data(DEFAULT_FILE)
    st.sidebar.success(f"✅ 自动加载: {DEFAULT_FILE}")
elif os.path.exists(ALT_FILE):
    df = load_and_process_data(ALT_FILE)
    st.sidebar.success(f"✅ 自动加载: {ALT_FILE}")

if st.sidebar.checkbox("上传新文件覆盖 (或手动上传)"):
    uploaded_file = st.sidebar.file_uploader("上传 CSV", type=['csv'])
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)

if df is not None:
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
            "图7: 参赛场次与效率": plot_fig7,
            "图16: 接球能力与综合表现": plot_fig16
        },
        "球员特写": {
            "图3: Virat Kohli 年度趋势": plot_fig3,
            "图5: 顶级球员雷达图": plot_fig5
        }
    }
    
    st.sidebar.header("📊 图表导航")
    category = st.sidebar.selectbox("选择分析维度", list(chart_map.keys()))
    chart_name = st.sidebar.radio("选择图表", list(chart_map[category].keys()))
    
    st.subheader(f"📈 {chart_name}")
    try:
        fig = chart_map[category][chart_name](df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"图表生成失败: {e}")
        st.write("请检查数据文件是否正确")
else:
    st.info("👋 请上传数据文件 data.csv 以开始分析")