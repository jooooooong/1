import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

st.title("📍 배송 위치 자동 군집 분석 (Folium 지도 시각화)")

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Delivery.csv")

df = load_data()
st.subheader("📄 데이터 미리보기")
st.dataframe(df)

# 위치 컬럼 지정
lat_col = "Latitude"
lon_col = "Longitude"

if lat_col not in df.columns or lon_col not in df.columns:
    st.error("위치 정보가 누락되었습니다 (Latitude / Longitude 필요).")
    st.stop()

# 군집 수 조절
st.sidebar.header("⚙️ 군집 분석 설정")
n_clusters = st.sidebar.slider("군집 수 (K)", min_value=2, max_value=10, value=3)

# 데이터 전처리
X = df[[lat_col, lon_col]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 군집 분석
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)
X_result = df.loc[X.index].copy()
X_result["Cluster"] = labels

# 중심 위치
center_lat = X_result[lat_col].mean()
center_lon = X_result[lon_col].mean()

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
colors = [
    "red", "blue", "green", "purple", "orange", "darkred", 
    "lightblue", "pink", "gray", "cadetblue"
]

for _, row in X_result.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=5,
        color=colors[int(row["Cluster"]) % len(colors)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster {row['Cluster']}"
    ).add_to(m)

st.subheader("🌍 군집 결과 지도")
st_folium(m, width=700, height=500)
