import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Data prepering

data = pd.read_csv('Dataset_4f_time_only/clustering_results_4c_4f.csv', header=0)
data = data.copy()
data = data.drop(columns=['mediana_czas','max_czas'])
data = data.fillna(0)

def feature_mish(top, seq):
    top_tokens=re.findall(r'[+-][pl]|d', str(top))
    seq_tokens=list(str(seq))

    if len(top_tokens)==len(seq_tokens):
        combined_toc= [t+s for t,s in zip(top_tokens,seq_tokens)]
        return "".join(combined_toc)

    else:
        return None

data['combined_toc']=data.apply(lambda row: feature_mish(row['top'], row['seq']), axis=1)
surowka=data['combined_toc'].tolist()

# model do embendingu

ft_model=FastText(sentences=surowka, vector_size=8, window=3, min_count=1, min_n=2, max_n=3)
ft_model.save('model_embedding.bin')
#ft_model=FastText.load('Database/model_embedding.bin')


words = ['-p1', '-p2', '-p3', '+l1', '+l2', '+l3', 'd']
vectors = [ft_model.wv[w] for w in words]

# Squash to 2D
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(8,6))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.annotate(word, (coords[i, 0], coords[i, 1]))

plt.title("How the AI sees your Loop 'Closeness'")
#plt.show()


def get_sequence_embedding(tokens, model, expected_loops=3):
    vector_dim = model.vector_size
    feature_vectors = []

    # Get vectors for the tokens we have
    for t in tokens:
        if t in model.wv:
            feature_vectors.append(model.wv[t])
        else:
            # If a token is missing from the model, use zeros
            feature_vectors.append(np.zeros(vector_dim))

    # FORCED PADDING/TRUNCATING
    # If the row is too short, add zero vectors
    while len(feature_vectors) < expected_loops:
        feature_vectors.append(np.zeros(vector_dim))

    # If the row is too long, cut it off
    feature_vectors = feature_vectors[:expected_loops]

    # Flatten into a single 1D array
    return np.concatenate(feature_vectors)

X = np.array([get_sequence_embedding(t, ft_model) for t in data['combined_toc']])
X=pd.DataFrame(X)
X['cluster_results_KMeans']=data['cluster_results_KMeans']
X['cluster_results_KMedoids']=data['cluster_results_KMedoids']
X.to_csv('vector_data_4f_4c_time_only.csv', index=False)