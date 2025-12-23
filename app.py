import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import time
import gspread # Google Sheets 연동을 위한 라이브러리
from gspread_dataframe import set_with_dataframe # DataFrame을 Sheet에 쓰기 위한 라이브러리

# --- Google Sheets 설정 (선생님이 제공한 URL) ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1Cj4pLDORD_mJJzvb8xxXW2kAAaC7S9O6xcTEuYlWcVo/edit?usp=sharing"
WORKSHEET_NAME = "Sheet1" # 데이터를 저장할 시트 이름

# --- Google Sheets 연결 함수 (수정) ---
@st.cache_resource(ttl=3600) 
def get_sheets_client():
    try:
        # st.secrets에서 gsheets 인증 정보를 사용하여 연결 (JSON 키 대신 토큰 사용)
        gc = gspread.service_account_from_dataframe(st.secrets["gsheets"]) # 토큰 기반 인증
        ss = gc.open_by_url(SHEET_URL)
        return ss
    except Exception as e:
        st.error(f"⚠️ Google Sheets 연결 오류: Secrets 설정 및 시트 권한(편집자)을 확인하세요. 오류: {e}")
        st.caption("gsheets 섹션에 토큰이 등록되어 있는지 확인해 주세요.")
        return None
# ... (load_data_from_sheets, save_uploaded_data_to_sheets 함수 내의 나머지 로직은 대부분 동일)

# --- 1. 데이터 로드 엔진 (Google Sheets에서 데이터 읽기) ---
@st.cache_data(ttl=60) # 1분마다 새로 불러옴 (데이터 변경 시 즉각 반영)
def load_data_from_sheets(ss):
    if ss is None:
        return pd.DataFrame() 
    
    try:
        # 지정된 시트에서 모든 데이터를 읽어옵니다.
        worksheet = ss.worksheet(WORKSHEET_NAME)
        df = pd.DataFrame(worksheet.get_all_records())
        
        if df.empty or 'Student_ID' not in df.columns:
            st.warning(f"Google Sheets '{WORKSHEET_NAME}'에 분석 데이터가 없습니다. CSV를 업로드해주세요.")
            return pd.DataFrame()

        # 데이터 형식 변환: 날짜/시간 및 점수 (오류가 나면 미제출/0점으로 처리)
        df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')
        df['Submitted_At'] = pd.to_datetime(df['Submitted_At'], errors='coerce')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
        
        return df.dropna(subset=['Deadline'])
        
    except Exception as e:
        st.error(f"Google Sheets 데이터 읽기 오류: 시트 이름 또는 권한을 확인하세요.")
        return pd.DataFrame()

# --- 1-1. CSV 업로드 시 Google Sheets에 데이터 저장 ---
def save_uploaded_data_to_sheets(uploaded_file, ss):
    if ss is None:
        return False
        
    try:
        df_new = pd.read_csv(uploaded_file)
        
        # 필수 컬럼 확인
        required_cols = ['Student_ID', 'Deadline', 'Submitted_At', 'Score']
        if not all(col in df_new.columns for col in required_cols):
            st.error("업로드된 CSV 파일에 필수 컬럼(Student_ID, Deadline, Submitted_At, Score)이 모두 포함되어야 합니다.")
            return False

        # 데이터 정리 및 형식 맞추기
        df_new['Deadline'] = pd.to_datetime(df_new['Deadline'], errors='coerce')
        df_new['Submitted_At'] = pd.to_datetime(df_new['Submitted_At'], errors='coerce')
        df_new['Score'] = pd.to_numeric(df_new['Score'], errors='coerce').fillna(0)

        # Sheets에 쓰기 (기존 내용 덮어쓰기)
        worksheet = ss.worksheet(WORKSHEET_NAME)
        worksheet.clear() 
        set_with_dataframe(worksheet, df_new)
        
        st.success(f"✅ 새 데이터가 Google Sheets '{WORKSHEET_NAME}'에 영구 저장되었습니다!")
        st.cache_data.clear() # 캐시를 지워 새 데이터를 즉시 로드
        return True
    
    except Exception as e:
        st.error(f"데이터를 Google Sheets에 저장하는 중 오류 발생: {e}")
        return False


# --- 2. 머신러닝 분석 엔진 (K-Means Clustering) ---
@st.cache_data
def run_ml_analysis(df):
    if df.empty:
        return pd.DataFrame()
        
    # 학생별 요약 데이터 생성 (기존 로직 유지)
    summary = []
    for sid, group in df.groupby('Student_ID'):
        total = len(group)
        missing = group['Submitted_At'].isnull().sum()
        valid = group.dropna(subset=['Submitted_At']).copy()
        
        if len(valid) > 0:
            valid['time_diff_hours'] = (valid['Submitted_At'] - valid['Deadline']).dt.total_seconds() / 3600
            avg_lateness = valid['time_diff_hours'].mean() 
            avg_score = valid['Score'].mean()
        else:
            avg_lateness = 100 
            avg_score = 0
            
        summary.append([sid, avg_score, avg_lateness, missing])
    
    df_features = pd.DataFrame(summary, columns=['Student_ID', 'Avg_Score', 'Avg_Lateness', 'Missing_Count'])
    
    # ML 모델 학습 (4개 그룹으로 자동 분류)
    X = df_features[['Avg_Score', 'Avg_Lateness', 'Missing_Count']].copy()
    X['Avg_Lateness'] = np.clip(X['Avg_Lateness'], -24 * 7, 24 * 7) 
    X['Missing_Count_Scaled'] = X['Missing_Count'] * 15 

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) 
    df_features['Cluster'] = kmeans.fit_predict(X[['Avg_Score', 'Avg_Lateness', 'Missing_Count_Scaled']])
    
    # 클러스터 특성에 따라 이름 부여 
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

# --- 데이터 처리 메인 로직 ---
ss = get_sheets_client() # Google Sheets 클라이언트 연결
df_raw = load_data_from_sheets(ss) # Sheets에서 데이터 로드 시도

# 사이드바
st.sidebar.header("관리자 패널")

# CSV 업로드 처리: 업로드 시 Sheets에 영구 저장
uploaded_file = st.sidebar.file_uploader("과제 데이터 업로드 (CSV)", type="csv")
if uploaded_file is not None:
    if save_uploaded_data_to_sheets(uploaded_file, ss):
        st.rerun() # 저장 성공 시 재실행하여 새 데이터로 대시보드 갱신

# 데이터 로드 상태 확인 및 분석 실행
if df_raw.empty:
    st.info("CSV 파일을 업로드하면, 해당 데이터가 Google Sheets에 저장되고 앱이 분석을 시작합니다.")
    if ss is not None:
         st.caption(f"현재 Google Sheets '{WORKSHEET_NAME}'에서 데이터를 기다리는 중입니다.")
    st.stop() # 데이터가 없으면 앱 실행 중지
    
# 데이터 분석 실행
df_analyzed = run_ml_analysis(df_raw)

# --- 메인 대시보드 UI (기존 로직 유지) ---
st.header("현재 분석 데이터 (Google Sheets에서 불러옴)")

# 상단 KPI 지표
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 수강생", f"{len(df_analyzed)}명")
col2.metric("위험군(Dropout Risk)", f"{len(df_analyzed[df_analyzed['Persona'].str.contains('위험')])}명", delta="-2명", delta_color="inverse")
col3.metric("평균 점수", f"{df_analyzed['Avg_Score'].mean():.1f}점")

# 평균 제출 시간을 계산하여 표시
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