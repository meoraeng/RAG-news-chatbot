import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# macOS에서 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# [표 4] 데이터
data = {
    '프롬프트 전략': ['기본(Basic)', 'Few-shot', 'CoT', 'Self-Correction'],
    '정확성(Correctness)': [0.82, 0.85, 0.91, 0.93],
    '답변 관련성(Relevance)': [0.85, 0.88, 0.92, 0.95],
    'LLM-기반 포함률(%)': [85.00, 87.50, 81.80, 91.50],
    '키워드 포함률(%)': [25.41, 28.17, 32.55, 35.12],
    '평균 답변 길이(char)': [185, 230, 285, 310]
}
df = pd.DataFrame(data)

# --- 4.3.1 용 그래프: 프롬프트 전략의 고도화와 성능 향상 ---
def plot_overall_performance_trend(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['프롬프트 전략'], df['정확성(Correctness)'], marker='o', linestyle='-', label='정확성 (Correctness)', color='royalblue', lw=2)
    ax.plot(df['프롬프트 전략'], df['답변 관련성(Relevance)'], marker='o', linestyle='-', label='답변 관련성 (Relevance)', color='tomato', lw=2)
    ax.set_ylabel('점수')
    ax.set_title('[그림 6] 프롬프트 전략 고도화에 따른 성능 향상 추세', fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0.8, 1.0)
    ax.legend()
    fig.tight_layout()
    plt.savefig('graph_4_3_1_performance_trend.png', dpi=300)
    print("'graph_4_3_1_performance_trend.png' 파일이 저장되었습니다.")

# --- 4.3.2 용 그래프: CoT(Chain-of-Thought) 전략의 양면성 ---
def plot_cot_tradeoff(df):
    df_subset = df[df['프롬프트 전략'].isin(['Few-shot', 'CoT'])].copy()
    # 비교를 위해 관련성 점수를 0-100 스케일로 변환
    df_subset['답변 관련성(%)'] = df_subset['답변 관련성(Relevance)'] * 100

    labels = df_subset['프롬프트 전략']
    relevance_scores = df_subset['답변 관련성(%)']
    inclusion_scores = df_subset['LLM-기반 포함률(%)']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, relevance_scores, width, label='답변 관련성 (%)', color='skyblue')
    rects2 = ax.bar(x + width/2, inclusion_scores, width, label='LLM-기반 포함률 (%)', color='salmon')

    ax.set_ylabel('점수 (%)')
    ax.set_title('[그림 7] CoT 전략의 성능-완전성 트레이드오프', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(75, 100)
    ax.legend()

    ax.bar_label(rects1, fmt='%.1f%%', padding=3)
    ax.bar_label(rects2, fmt='%.1f%%', padding=3)

    fig.tight_layout()
    plt.savefig('graph_4_3_2_cot_tradeoff.png', dpi=300)
    print("'graph_4_3_2_cot_tradeoff.png' 파일이 저장되었습니다.")

# --- 4.3.3 용 그래프: Self-Correction 전략의 효과 검증 ---
def plot_self_correction_effect(df):
    df_subset = df[df['프롬프트 전략'].isin(['CoT', 'Self-Correction'])]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['salmon', 'mediumseagreen']
    bars = ax.bar(df_subset['프롬프트 전략'], df_subset['LLM-기반 포함률(%)'], color=colors)

    ax.set_ylabel('LLM-기반 포함률 (%)')
    ax.set_title('[그림 8] Self-Correction의 답변 완전성 개선 효과', fontsize=14, pad=15)
    ax.set_ylim(80, 100)
    
    # 개선 수치와 화살표 추가 (개선된 버전)
    cot_val = df_subset.iloc[0]['LLM-기반 포함률(%)']
    sc_val = df_subset.iloc[1]['LLM-기반 포함률(%)']
    improvement = sc_val - cot_val

    # 화살표 그리기 (시작: CoT 바 상단, 끝: Self-Correction 바 상단)
    ax.annotate('',
                xy=(1, sc_val),
                xytext=(0, cot_val),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.2",
                    color='black',
                    lw=1.5
                ))
                
    # 개선 수치 텍스트 추가 (화살표 중앙 상단)
    mid_x = 0.5
    mid_y = (cot_val + sc_val) / 2
    ax.text(mid_x, mid_y + 1, f'+{improvement:.2f}%p',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='#006400') # 짙은 녹색

    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    fig.tight_layout()
    plt.savefig('graph_4_3_3_self_correction_effect.png', dpi=300)
    print("'graph_4_3_3_self_correction_effect.png' 파일이 저장되었습니다.")

if __name__ == '__main__':
    plot_overall_performance_trend(df.copy())
    plot_cot_tradeoff(df.copy())
    plot_self_correction_effect(df.copy())
    plt.show() 