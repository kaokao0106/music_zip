import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


st.title('楽曲レコメンド')

def load_data_and_model():
#データの読み込み
    music = pd.read_csv("data_with_clusters.csv")
    music['name_low'] = music['name'].str.lower()

    
    #スケーラーの読み込み
    scaler = StandardScaler()
    scaler.mean_ = np.load("scaler_mean.npy")
    scaler.scale_ = np.load("scaler_scale.npy")
    
    #KMeansクラスタ中心の読み込み
    kmeans_centroids = np.load("kmeans_centroids.npy")
    kmeans = KMeans(n_clusters=kmeans_centroids.shape[0])
    kmeans.cluster_centers_ = kmeans_centroids
    
    return music, scaler, kmeans

music, scaler, kmeans = load_data_and_model()
#アルファベット順にソート
song_names = sorted(music['name'].unique().tolist())
#使用する特徴量
audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]

#ユーザーインターフェース
st.markdown("## この楽曲に似ている曲を探す")
selected_song = st.selectbox("楽曲を選んでください", song_names)
#選ばれた曲名と一致する行
song_row = music[music['name_low'] == selected_song.lower()]

if song_row.empty:
    st.error(f"選択された曲 '{selected_song}' はデータに存在しません。")
else:
    song = song_row.iloc[0]
    st.write(f"あなたが選んだ曲: **{song['name']}** by {song['artists']} ({song['year']})")


#似ている楽曲を表示
st.markdown(f"### {song['name']}に似ている楽曲")

#その曲（song）のクラスタを取得
cluster_label = song['cluster_label']
#そのクラスタに属する曲だけを抽出
cluster_songs = music[music['cluster_label'] == cluster_label]

#自分が選択している楽曲は除外する
cluster_songs = cluster_songs[cluster_songs['name'] != song['name']].copy()

#特徴量のそれぞれのデータをスケーリング（標準化）する
#選ばれた一曲（基準）
song_vector = scaler.transform(song[audio_features].values.reshape(1, -1))
#同じクラスタ内の曲たち（比較対象）
cluster_vectors = scaler.transform(cluster_songs[audio_features])

#ユークリッド距離（似ている距離）で近い順に並べる
distances = cdist(song_vector, cluster_vectors, metric='euclidean')[0]
#距離をデータフレームに追加
cluster_songs['distance'] = distances
#距離が小さい順に並べて上位５位を取得する
recommendations = cluster_songs.sort_values('distance').head(5)


for i, row in recommendations.iterrows():
        st.write(f"🎵 {row['name']} by {row['artists']} ({row['year']})")
        st.write(f"  類似度スコア（距離）: {row['distance']:.3f}")
        st.write("---")

