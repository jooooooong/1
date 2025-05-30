import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# 데이터 로드
@st.cache_data
def load_data():
    return pd.read_csv("Delivery (1).csv")

df = load_data()

st.title("📦 배송 클러스터링 & 지도 시각화")

# 변수 고정: 위도/경도
lat_col = "Latitude"
lon_col = "Longitude"

# 수치형 변수 선택
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_cols = st.multiselect("군집에 사용할 변수 선택", options=numeric_cols, default=["Latitude", "Longitude"])

if len(selected_cols) < 2:
    st.warning("최소 2개 이상의 수치형 변수를 선택해주세요.")
    st.stop()

# 클러스터 수 선택
k = st.slider("클러스터 수 선택 (K)", 2, 10, 3)

# 군집 수행
X = df[selected_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_scaled)

# PCA (2차원 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_vis = df.loc[X.index].copy()
df_vis["Cluster"] = labels
df_vis["PCA1"] = X_pca[:, 0]
df_vis["PCA2"] = X_pca[:, 1]

# Plotly 시각화
st.subheader("📊 PCA 기반 군집 시각화")
fig = px.scatter(df_vis, x="PCA1", y="PCA2", color=df_vis["Cluster"].astype(str),
                 title="K-Means 클러스터링 결과 (PCA 2D)",
                 color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig, use_container_width=True)

# 지도 시각화
st.subheader("🗺️ 지도 기반 클러스터 시각화")
map_center = [df_vis[lat_col].mean(), df_vis[lon_col].mean()]
m = folium.Map(location=map_center, zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'pink', 'gray', 'black']

for _, row in df_vis.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=5,
        color=colors[row['Cluster'] % len(colors)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster: {row['Cluster']}<br>Lat: {row[lat_col]}<br>Lon: {row[lon_col]}"
    ).add_to(marker_cluster)

st_folium(m, width=700)
pip install streamlit folium streamlit-folium scikit-learn pandas plotly
streamlit run app.py

