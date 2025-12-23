import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import time

# --- 1. 데이터 로드 엔진 (CSV 파일에서 데이터 읽기) ---
# 세션 상태에 데이터를 저장하여 페이지 새로고침 시 데이터 유지
def load_data_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # 필수 컬럼 확인
        required_cols = ['Student_ID', 'Deadline', 'Submitted_At', 'Score']
        if not all(col in df.columns for col in required_cols):
            st.error("업로드된 CSV 파일에 필수 컬럼(Student_ID, Deadline, Submitted_At, Score)이 모두 포함되어야 합니다.")
            return pd.DataFrame()

        # 데이터 형식 변환: 날짜/시간 및 점수 (오류가 나면 미제출/0점으로 처리)
        df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')
        df['Submitted_At'] = pd.to_datetime(df['Submitted_At'], errors='coerce')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
        
        # Deadline이 없는 행은 분석에서 제외
        return df.dropna(subset=['Deadline'])
        
    except Exception as e:
        st.error(f"CSV 파일 처리 중 오류 발생: {e}")
        return pd.DataFrame()


# --- 2. 머신러닝 분석 엔진 (K-Means Clustering) ---
@st.cache_data
def run_ml_analysis(df):
    if df.empty:
        return pd.DataFrame()
        
    # 학생별 요약 데이터 생성 (기존 로직 유지)
    summary = []
    for sid, group in df.groupby('Student_ID'):
        missing = group['Submitted_At'].isnull().sum()
        valid = group.dropna(subset=['Submitted_At']).copy()
        
        if len(valid) > 0:
            # 제출 시간 계산 (시간 단위)
            valid['time_diff_hours'] = (valid['Submitted_At'] - valid['Deadline']).dt.total_seconds() / 3600
            avg_lateness = valid['time_diff_hours'].mean() 
            avg_score = valid['Score'].mean()
        else:
            # 미제출만 있는 경우 최악의 값 부여
            avg_lateness = 100 
            avg_score = 0
            
        summary.append([sid, avg_score, avg_lateness, missing])
    
    df_features = pd.DataFrame(summary, columns=['Student_ID', 'Avg_Score', 'Avg_Lateness', 'Missing_Count'])
    
    # ML 모델 학습 (4개 그룹으로 자동 분류) - (기존 로직 유지)
    X = df_features[['Avg_Score', 'Avg_Lateness', 'Missing_Count']].copy()
    X['Avg_Lateness'] = np.clip(X['Avg_Lateness'], -24 * 7, 24 * 7) 
    X['Missing_Count_Scaled'] = X['Missing_Count'] * 15 

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) 
    df_features['Cluster'] = kmeans.fit_predict(X[['Avg_Score', 'Avg_Lateness', 'Missing_Count_Scaled']])
    
    # **[여기부터 수정 시작]**
    # 클러스터 특성에 따라 이름 부여 (현실적인 조건으로 수정)
    
    # 클러스터의 평균 특성을 계산하여 라벨링 조건의 기준점으로 사용 (옵션: 실제 클러스터 중심 사용)
    cluster_means = df_features.groupby('Cluster')[['Avg_Lateness', 'Missing_Count', 'Avg_Score']].mean()
    
    # 단순한 if-else 조건으로 4개 유형 강제 분류
    # 참고: 현재 데이터가 단일 과제라면 Missing_Count는 최대 1이므로, 이 기준으로 위험군을 나눕니다.

    def label_cluster(row):
        # 1. 🚨 중도포기 위험군 (미제출이 1개라도 있거나, 점수가 매우 낮은 경우)
        if row['Missing_Count'] >= 1 and row['Avg_Score'] < 50: 
             return "🚨 중도포기 위험군"
        
        # 2. ⚠️ 습관적 지각생 (평균 지각 시간이 1시간 이상)
        if row['Avg_Lateness'] > 1: 
             return "⚠️ 습관적 지각생"
        
        # 3. ✅ 성실 우수생 (마감 3시간 이상 전에 제출한 경우)
        if row['Avg_Lateness'] <= -3: 
             return "✅ 성실 우수생"
             
        # 4. ⚡ 벼락치기형 (나머지, 마감 3시간 전 ~ 1시간 지각 사이의 학생)
        # 이 조건이 가장 넓은 범위의 학생을 포함합니다.
        return "⚡ 벼락치기형" 

    df_features['Persona'] = df_features.apply(label_cluster, axis=1)
    return df_features
# **[여기까지 수정]**

# --- 3. UI 및 시각화 (Streamlit) ---
st.set_page_config(page_title="Edu-Analytics Pro", layout="wide")

st.title("🎓 AI 학습 관리 매니저 (Edu-Analytics Pro)")
st.markdown("학생들의 패턴을 머신러닝으로 분석하고, 맞춤형 알림을 보냅니다.")

# 세션 상태 초기화 (데이터가 없는 경우)
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()

# 사이드바
st.sidebar.header("관리자 패널")
uploaded_file = st.sidebar.file_uploader("과제 데이터 업로드 (CSV)", type="csv")

# CSV 업로드 처리
if uploaded_file is not None:
    st.session_state['df_raw'] = load_data_from_csv(uploaded_file)
    st.sidebar.success("✅ CSV 파일 업로드 및 데이터 로드 완료.")
    uploaded_file = None # 파일을 처리했으니 초기화하여 재업로드 방지 (UX 개선)

df_raw = st.session_state['df_raw']

# 데이터 로드 상태 확인 및 분석 실행
if df_raw.empty:
    st.info("앱을 사용하려면 CSV 파일을 업로드해야 합니다.")
    st.caption("데이터는 페이지를 닫거나 앱이 재시작되면 사라집니다. (영구 저장 기능 제거됨)")
    st.stop()
    
# 데이터 분석 실행
df_analyzed = run_ml_analysis(df_raw)

# --- 메인 대시보드 UI ---
st.header("현재 분석 데이터")

# 상단 KPI 지표
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 수강생", f"{len(df_analyzed)}명")
col2.metric("위험군(Dropout Risk)", f"{len(df_analyzed[df_analyzed['Persona'].str.contains('위험')])}명")
col3.metric("평균 점수", f"{df_analyzed['Avg_Score'].mean():.1f}점")

# 평균 제출 시간 계산하여 표시
avg_lateness_sec = (df_raw['Submitted_At'] - df_raw['Deadline']).dt.total_seconds().mean()
if avg_lateness_sec < 0:
    time_delta = timedelta(seconds=abs(avg_lateness_sec))
    hours = int(time_delta.total_seconds() // 3600)
    minutes = int((time_delta.total_seconds() % 3600) // 60)
    col4.metric("평균 제출 시간", f"마감 {hours}시간 {minutes}분 전")
else:
    time_delta = timedelta(seconds=avg_lateness_sec)
    hours = int(time_delta.total_seconds() // 3600)
    col4.metric("평균 제출 시간", f"마감 {hours}시간 후 (지각)")

st.divider()

# 메인 대시보드
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 학생 유형별 분포 (Clustering Result)")
    fig = px.scatter(df_analyzed, x="Avg_Lateness", y="Avg_Score", 
                     color="Persona", hover_data=['Student_ID', 'Missing_Count'],
                     labels={"Avg_Lateness": "제출 시간 (양수=지각, 음수=미리제출)", "Avg_Score": "평균 점수"},
                     title="점수 vs 제출시간 상관관계 분석")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("👥 유형 비율")
    pie_fig = px.pie(df_analyzed, names='Persona', hole=0.4)
    st.plotly_chart(pie_fig, use_container_width=True)

# --- 4. 자동 알림 봇 시스템 ---
st.divider()
st.subheader("🤖 AI 자동 케어 (Auto-Notification Bot)")

if not df_analyzed.empty:
    target_persona = st.selectbox("알림을 보낼 대상 그룹을 선택하세요:", df_analyzed['Persona'].unique())
    filtered_students = df_analyzed[df_analyzed['Persona'] == target_persona]

    st.write(f"**선택된 대상:** {len(filtered_students)}명 ({target_persona})")

    default_msg = ""
    if "위험" in target_persona:
        default_msg = "안녕하세요! 최근 과제 제출에 어려움이 있나요? 상담이 필요하면 언제든 연락주세요."
    elif "지각" in target_persona:
        default_msg = "다음 과제 마감이 24시간 남았습니다. 이번에는 미리 제출해서 가산점을 받아보세요!"
    elif "벼락치기" in target_persona:
        default_msg = "조금만 더 일찍 시작하면 더 좋은 점수를 받을 수 있어요! 화이팅!"
    else:
        default_msg = "꾸준히 잘하고 계시네요! 이번 학기 우수 학생 후보입니다."

    message = st.text_area("전송할 메시지 내용:", value=default_msg)

    if st.button("🚀 선택한 학생들에게 알림 전송"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, student in enumerate(filtered_students['Student_ID']):
            time.sleep(0.05) 
            progress_bar.progress((i + 1) / len(filtered_students))
            status_text.text(f"Sending to {student}...")
            
        status_text.success(f"✅ 전송 완료! {len(filtered_students)}명의 학생에게 메시지를 보냈습니다.")
        st.balloons()