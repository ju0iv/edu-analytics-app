import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import time

# --- 1. 데이터 생성 엔진 (실제 앱에서는 파일 업로드로 대체) ---
@st.cache_data
def load_data():
    # 이전과 동일한 로직으로 가상 데이터 50명 생성
    num_students = 50
    data = []
    student_ids = [f'S{i:03d}' for i in range(1, num_students + 1)]
    base_deadline = datetime.now()
    
    for student in student_ids:
        # 0:성실, 1:벼락치기, 2:지각, 3:포기
        persona = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        
        for i in range(5):
            deadline = base_deadline - timedelta(days=(5 - i) * 7)
            
            if persona == 0: 
                submit_time = deadline - timedelta(days=np.random.randint(1, 4))
                score = np.random.randint(85, 100)
            elif persona == 1: 
                submit_time = deadline - timedelta(hours=np.random.randint(1, 5))
                score = np.random.randint(65, 90)
            elif persona == 2: 
                submit_time = deadline + timedelta(hours=np.random.randint(1, 48))
                score = np.random.randint(50, 75)
            else: 
                if np.random.random() > 0.6:
                    submit_time = None
                    score = 0
                else:
                    submit_time = deadline + timedelta(days=np.random.randint(2, 6))
                    score = np.random.randint(20, 50)
            
            data.append({
                'Student_ID': student,
                'Assignment': f'Week_{i+1}',
                'Deadline': deadline,
                'Submitted_At': submit_time,
                'Score': score
            })
    return pd.DataFrame(data)

# --- 2. 머신러닝 분석 엔진 (K-Means Clustering) ---
def run_ml_analysis(df):
    # 학생별 요약 데이터 생성
    summary = []
    for sid, group in df.groupby('Student_ID'):
        total = len(group)
        missing = group['Submitted_At'].isnull().sum()
        valid = group.dropna(subset=['Submitted_At']).copy()
        
        if len(valid) > 0:
            valid['time_diff_hours'] = (valid['Submitted_At'] - valid['Deadline']).dt.total_seconds() / 3600
            avg_lateness = valid['time_diff_hours'].mean() # 양수면 지각, 음수면 미리 제출
            avg_score = valid['Score'].mean()
        else:
            avg_lateness = 100 # 매우 늦음 처리
            avg_score = 0
            
        summary.append([sid, avg_score, avg_lateness, missing])
    
    df_features = pd.DataFrame(summary, columns=['Student_ID', 'Avg_Score', 'Avg_Lateness', 'Missing_Count'])
    
    # ML 모델 학습 (3개 그룹으로 자동 분류)
    # 실제로는 스케일링(StandardScaler)이 필요하지만 간단한 예시를 위해 생략
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_features['Cluster'] = kmeans.fit_predict(df_features[['Avg_Score', 'Avg_Lateness']])
    
    # 클러스터 특성에 따라 이름 부여 (Centroid 분석 기반 매핑 로직 필요하나 여기선 편의상 점수/지각도로 매핑)
    # 실제로는 클러스터 중심점을 보고 라벨링을 자동화해야 합니다.
    def label_cluster(row):
        if row['Missing_Count'] >= 2: return "🚨 중도포기 위험군"
        if row['Avg_Lateness'] > 0: return "⚠️ 습관적 지각생"
        if row['Avg_Lateness'] > -5 and row['Avg_Lateness'] <= 0: return "⚡ 벼락치기형"
        return "✅ 성실 우수생"

    df_features['Persona'] = df_features.apply(label_cluster, axis=1)
    return df_features

# --- 3. UI 및 시각화 (Streamlit) ---
st.set_page_config(page_title="Edu-Analytics Pro", layout="wide")

st.title("🎓 AI 학습 관리 매니저 (Edu-Analytics Pro)")
st.markdown("학생들의 패턴을 머신러닝으로 분석하고, 맞춤형 알림을 보냅니다.")

# 사이드바
st.sidebar.header("관리자 패널")
uploaded_file = st.sidebar.file_uploader("과제 데이터 업로드 (CSV)", type="csv")
if uploaded_file is None:
    st.sidebar.info("테스트용 가상 데이터를 사용합니다.")
    df_raw = load_data()
else:
    df_raw = pd.read_csv(uploaded_file)

# 데이터 분석 실행
df_analyzed = run_ml_analysis(df_raw)

# 상단 KPI 지표
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 수강생", f"{len(df_analyzed)}명")
col2.metric("위험군(Dropout Risk)", f"{len(df_analyzed[df_analyzed['Persona'].str.contains('위험')])}명", delta="-2명", delta_color="inverse")
col3.metric("평균 점수", f"{df_analyzed['Avg_Score'].mean():.1f}점")
col4.metric("평균 제출 시간", "마감 4시간 전")

st.divider()

# 메인 대시보드
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 학생 유형별 분포 (Clustering Result)")
    fig = px.scatter(df_analyzed, x="Avg_Lateness", y="Avg_Score", 
                     color="Persona", hover_data=['Student_ID'],
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

target_persona = st.selectbox("알림을 보낼 대상 그룹을 선택하세요:", df_analyzed['Persona'].unique())
filtered_students = df_analyzed[df_analyzed['Persona'] == target_persona]

st.write(f"**선택된 대상:** {len(filtered_students)}명 ({target_persona})")

# 메시지 템플릿 추천
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
        # 실제 앱에서는 여기에 KakaoTalk / Slack API 연동 코드가 들어갑니다.
        # 예: send_kakao_message(student_id, message)
        time.sleep(0.05) # 전송 시간 시뮬레이션
        progress_bar.progress((i + 1) / len(filtered_students))
        status_text.text(f"Sending to {student}...")
        
    status_text.success(f"✅ 전송 완료! {len(filtered_students)}명의 학생에게 메시지를 보냈습니다.")
    st.balloons()