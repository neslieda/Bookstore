import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import re
import warnings

warnings.filterwarnings('ignore')

def preprocess_text(text):

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]

    return tokens


def create_word2vec_model(texts, vector_size=300):
    processed_texts = [preprocess_text(text) for text in texts]
    model = Word2Vec(sentences=processed_texts,
                     vector_size=vector_size,
                     window=5,
                     min_count=2,
                     workers=4)
    return model, processed_texts


def get_document_vector(text, model):
    tokens = preprocess_text(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def evaluate_clustering(X, labels):
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }
    return metrics


def plot_evaluation_metrics(metrics_dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clustering Evaluation Metrics')

    axes[0, 0].plot(metrics_dict['n_clusters'], metrics_dict['silhouette'], marker='o')
    axes[0, 0].set_title('Silhouette Score (Higher is better)')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Score')

    axes[0, 1].plot(metrics_dict['n_clusters'], metrics_dict['davies_bouldin'], marker='o')
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is better)')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Score')

    axes[1, 0].plot(metrics_dict['n_clusters'], metrics_dict['calinski_harabasz'], marker='o')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is better)')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Score')

    axes[1, 1].plot(metrics_dict['n_clusters'], metrics_dict['inertia'], marker='o')
    axes[1, 1].set_title('Elbow Method (Inertia)')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Inertia')

    plt.tight_layout()
    plt.savefig('BOOK_CLUSTER_METRICS.png')
    plt.close()


def plot_clusters(X_2d, labels, title):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('BOOK_CLUSTER.png')
    plt.close()


def find_optimal_clusters(X, max_clusters=10):
    metrics_dict = {
        'n_clusters': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []
    }

    for n_clusters in range(2, max_clusters + 1):
        print(f"Testing {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        metrics_dict['n_clusters'].append(n_clusters)
        metrics_dict['silhouette'].append(silhouette_score(X, labels))
        metrics_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
        metrics_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        metrics_dict['inertia'].append(kmeans.inertia_)

    return metrics_dict


def save_results(df, metrics, metrics_dict):
    df.to_csv('clustering_results_Word2Vec.csv', index=False)

    converted_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            converted_metrics[k] = float(v)
        else:
            converted_metrics[k] = v

    converted_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (list, np.ndarray)):
            converted_metrics_dict[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
        else:
            converted_metrics_dict[k] = v

    results = {
        'final_metrics': converted_metrics,
        'optimization_metrics': converted_metrics_dict
    }

    with open('clustering_results_Word2Vec.json', 'w') as f:
        json.dump(results, f, indent=2)


def get_cluster_descriptions(df):
    cluster_data = {}
    for cluster in df['Cluster'].unique():
        cluster_books = df[df['Cluster'] == cluster]
        cluster_data[f'cluster_{cluster}'] = {
            'size': len(cluster_books),
            'books': []
        }

        for _, book in cluster_books.iterrows():
            book_info = {
                'title': book['Title'] if 'Title' in book else 'Unknown',
                'description': book['Description']
            }
            # Add other available book fields if they exist
            for field in ['Author', 'Genre', 'Rating']:
                if field in book:
                    book_info[field.lower()] = book[field]

            cluster_data[f'cluster_{cluster}']['books'].append(book_info)

    return cluster_data


def save_results(df, metrics, metrics_dict):
    df.to_csv('clustering_results_Word2Vec.csv', index=False)
    cluster_data = get_cluster_descriptions(df)
    converted_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            converted_metrics[k] = float(v)
        else:
            converted_metrics[k] = v

    converted_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (list, np.ndarray)):
            converted_metrics_dict[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
        else:
            converted_metrics_dict[k] = v

    results = {
        'final_metrics': converted_metrics,
        'optimization_metrics': converted_metrics_dict,
        'clusters': cluster_data
    }

    with open('clustering_results_Word2Vec.json', 'w') as f:
        json.dump(results, f, indent=2)


def main(file_path, n_clusters=5, find_optimal=True):

    print("Loading data...")
    df = pd.read_csv(file_path)

    print("Creating Word2Vec model...")
    w2v_model, processed_texts = create_word2vec_model(df['Description'])

    print("Creating document vectors...")
    doc_vectors = np.array([get_document_vector(text, w2v_model)
                            for text in df['Description']])

    if find_optimal:
        print("Finding optimal number of clusters...")
        metrics_dict = find_optimal_clusters(doc_vectors)
        plot_evaluation_metrics(metrics_dict)
    else:
        metrics_dict = {}

    print(f"Performing final clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_vectors)

    metrics = evaluate_clustering(doc_vectors, cluster_labels)
    print("\nFinal Clustering Metrics:")
    print(f"Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f}")

    print("Creating visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    doc_vectors_2d = tsne.fit_transform(doc_vectors)
    plot_clusters(doc_vectors_2d, cluster_labels,
                  'Book Description Clusters Visualization')

    df['Cluster'] = cluster_labels

    print("Saving results...")
    save_results(df, metrics, metrics_dict)

    w2v_model.save("book_descriptions_word2vec.model")

    return df, metrics, metrics_dict


if __name__ == "__main__":
    df, metrics, metrics_dict = main("books.csv", n_clusters=5, find_optimal=True)