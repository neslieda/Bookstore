import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text, return_tokens=False):

    stop_words = set(stopwords.words('english') + stopwords.words('turkish'))
    lemmatizer = WordNetLemmatizer()

    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]

    return tokens if return_tokens else ' '.join(tokens)


def create_word2vec_model(documents, vector_size=300):

    tokens_list = [preprocess_text(doc, return_tokens=True) for doc in documents]
    w2v_model = Word2Vec(sentences=tokens_list,
                         vector_size=vector_size,
                         window=5,
                         min_count=2,
                         workers=4)

    doc_vectors = []
    for tokens in tokens_list:
        vector = np.zeros(vector_size)
        count = 0
        for token in tokens:
            if token in w2v_model.wv:
                vector += w2v_model.wv[token]
                count += 1
        if count > 0:
            vector /= count
        doc_vectors.append(vector)

    return np.array(doc_vectors), w2v_model


def get_cluster_keywords(w2v_model, doc_vectors, cluster_labels, n_keywords=10):

    cluster_keywords = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        cluster_vectors = doc_vectors[cluster_labels == cluster]
        cluster_center = cluster_vectors.mean(axis=0)

        similar_words = w2v_model.wv.similar_by_vector(cluster_center, topn=n_keywords)
        cluster_keywords[f"Küme {cluster}"] = [word for word, _ in similar_words]

    return cluster_keywords


def cluster_descriptions(file_path, method='dbscan', **kwargs):

    print("Veri yükleniyor...")
    df = pd.read_csv(file_path)
    df['Description'] = df['Description'].fillna('')
    df = df[df['Description'].str.len() > 50]

    print("Word2Vec modeli oluşturuluyor...")
    doc_vectors, w2v_model = create_word2vec_model(df['Description'])

    doc_vectors_normalized = normalize(doc_vectors)

    results = []

    if method in ['dbscan', 'all']:
        print("\nDBSCAN kümeleme yapılıyor...")
        dbscan = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5),
            metric='euclidean'
        )
        dbscan_labels = dbscan.fit_predict(doc_vectors_normalized)

        if len(np.unique(dbscan_labels)) > 1:
            dbscan_silhouette = silhouette_score(doc_vectors_normalized, dbscan_labels)
            dbscan_keywords = get_cluster_keywords(w2v_model, doc_vectors, dbscan_labels)
            results.append({
                'method': 'DBSCAN',
                'labels': dbscan_labels.tolist(),
                'silhouette': float(dbscan_silhouette),
                'keywords': dbscan_keywords
            })

    if method in ['hierarchical', 'all']:
        print("\nHierarchical kümeleme yapılıyor...")
        n_clusters = kwargs.get('n_clusters', 5)

        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average'
        )
        hierarchical_labels = hierarchical.fit_predict(doc_vectors_normalized)

        hierarchical_silhouette = silhouette_score(doc_vectors_normalized, hierarchical_labels)
        hierarchical_keywords = get_cluster_keywords(w2v_model, doc_vectors, hierarchical_labels)
        results.append({
            'method': 'Hierarchical',
            'labels': hierarchical_labels.tolist(),
            'silhouette': float(hierarchical_silhouette),
            'keywords': hierarchical_keywords
        })

    return df, results, doc_vectors


def save_results_to_json(df, results, doc_vectors, output_file='clustering_results.json'):

    output_data = {
        'data_info': {
            'total_documents': len(df),
            'columns': df.columns.tolist()
        },
        'clustering_results': []
    }

    for result in results:
        method_result = {
            'method': result['method'],
            'silhouette': result['silhouette'],
            'keywords': result['keywords'],
            'clusters': {}
        }

        labels = result['labels']
        unique_clusters = sorted(set(labels))

        for cluster in unique_clusters:
            if cluster == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster_{cluster}"

            cluster_indices = [i for i, label in enumerate(labels) if label == cluster]

            cluster_books = df.iloc[cluster_indices][['Title', 'Description']].to_dict('records')

            method_result['clusters'][cluster_name] = {
                'size': len(cluster_indices),
                'books': cluster_books,
                'keywords': result['keywords'].get(f"Küme {cluster}", []) if cluster != -1 else []
            }

        output_data['clustering_results'].append(method_result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")

if __name__ == "__main__":
    file_path = 'books.csv'

    df, results, doc_vectors = cluster_descriptions(
        file_path,
        method='all',
        eps=0.5,
        min_samples=5,
        n_clusters=5
    )

    save_results_to_json(df, results, doc_vectors)

if __name__ == "__main__":
    file_path = 'books.csv'

    df, results, doc_vectors = cluster_descriptions(
        file_path,
        method='all',
        eps=0.5,
        min_samples=5,
        n_clusters=5
    )
