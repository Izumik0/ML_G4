import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import plotly.express as px


# importy danych - tu można zmieniać
df=pd.read_csv('Database/Dataset_85/copy.csv')
etykiety=['sredni_czas', 'stddevi', 'obecnosc','max_czas']


df=df.fillna(0)
rzeczy=df[etykiety]
#normalizacja
norm=MinMaxScaler().fit_transform(rzeczy)

# tabela
tab=pd.DataFrame(norm, columns=etykiety)
#clustering - KMeans
cluster_KMeans=KMeans(n_clusters=4, random_state=42)
wynik_KMeans=cluster_KMeans.fit_predict(tab)
kolumny=['top','seq']+etykiety
zest=df[kolumny].copy()
zest['cluster_results_KMeans']=wynik_KMeans
fig_KMeans = px.scatter_3d(
    zest,
    x=etykiety[0],
    y=etykiety[1],
    z=etykiety[2],
    color='cluster_results_KMeans',  # This colors the points by cluster
    title='3D Cluster Visualization KMeans Method',
    labels={'cluster_results': 'Cluster ID'},
    opacity=0.7
)
#fig_KMeans.show()

# clustering - KMedoids

cluster_KMedoids=KMedoids(n_clusters=4, random_state=42)
wynik_KMedoids=cluster_KMedoids.fit_predict(tab)
zest['cluster_results_KMedoids']=wynik_KMedoids

fig_KMeans = px.scatter_3d(
    zest,
    x=etykiety[0],
    y=etykiety[1],
    z=etykiety[2],
    color='cluster_results_KMedoids',  # This colors the points by cluster
    title='3D Cluster Visualization KMedoids Method',
    labels={'cluster_results': 'Cluster ID'},
    opacity=0.7
)
#fig_KMeans.show()
print(zest.head())
print("Rozkład klas w KMeans:", zest['cluster_results_KMeans'].value_counts())
print("Rozkład klas w KMedoids:", zest['cluster_results_KMedoids'].value_counts())
zest.to_csv('clustering_results_4c_4f.csv', index=False)