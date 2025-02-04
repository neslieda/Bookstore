import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from umap import UMAP
import json
from tqdm import tqdm


class OptimizedBookClusterAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.df = None

    def preprocess_text(self, descriptions):
        processed_texts = []
        for text in descriptions:
            if pd.isna(text):
                processed_texts.append('')
                continue

            text = str(text).lower()
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
            text = ' '.join(text.split())
            processed_texts.append(text)

        return processed_texts

    def analyze_clusters(self, labels):
        unique_labels = np.unique(labels)
        clusters = {}

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.where(labels == label)[0]

            cluster_texts = self.df.iloc[cluster_indices].to_dict('records')

            clusters[f'cluster_{label}'] = {
                'size': len(cluster_texts),
                'books': cluster_texts
            }

        return clusters

    def cluster_descriptions(self, csv_path):
        try:
            print("Veri okunuyor ve işleniyor...")
            self.df = pd.read_csv(csv_path)
            descriptions = self.preprocess_text(self.df['Description'].tolist())

            print("BERT embeddings oluşturuluyor...")
            embeddings = self.model.encode(descriptions, show_progress_bar=True)

            print("UMAP ile boyut indirgeme yapılıyor...")
            reducer = UMAP(
                n_components=30,
                n_neighbors=30,
                min_dist=0.1,
                random_state=42
            )
            reduced_embeddings = reducer.fit_transform(embeddings)

            eps_values = [0.5, 0.7, 0.9, 1.1]
            min_samples_values = [10, 15, 20, 25]

            best_score = -1
            best_results = None
            best_labels = None

            print("DBSCAN kümeleme yapılıyor...")
            for eps in tqdm(eps_values):
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(reduced_embeddings)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_points = sum(labels == -1)

                    if n_clusters < 3 or n_clusters > 10:
                        continue

                    if noise_points > len(labels) * 0.3:
                        continue

                    valid_mask = labels != -1
                    if sum(valid_mask) < 2:
                        continue

                    silhouette_avg = silhouette_score(
                        reduced_embeddings[valid_mask],
                        labels[valid_mask]
                    )

                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_labels = labels
                        best_results = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'silhouette_score': float(silhouette_avg),
                            'n_clusters': n_clusters,
                            'noise_points': int(noise_points),
                            'noise_percentage': float(noise_points / len(labels) * 100)
                        }

            if best_results and best_labels is not None:
                best_results['clusters'] = self.analyze_clusters(best_labels)

                results = {
                    'clustering_results_Bert': best_results,
                    'metadata': {
                        'total_samples': len(descriptions),
                        'embedding_model': 'all-mpnet-base-v2',
                        'reduction_method': 'UMAP',
                        'clustering_method': 'DBSCAN',
                        'umap_params': {
                            'n_components': 30,
                            'n_neighbors': 30,
                            'min_dist': 0.1
                        }
                    }
                }

                with open('clustering_results_Bert.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                return results

            return None

        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = OptimizedBookClusterAnalyzer()
    results = analyzer.cluster_descriptions('books.csv')

    if results:
        clustering = results['clustering_results_Bert']
        print("\nKümeleme Sonuçları:")
        print(f"Silhouette score: {clustering['silhouette_score']:.4f}")
        print(f"Küme sayısı: {clustering['n_clusters']}")
        print(f"Parametreler: eps={clustering['eps']}, min_samples={clustering['min_samples']}")
        print(f"Gürültü noktaları: {clustering['noise_points']} ({clustering['noise_percentage']:.1f}%)")

        print("\nKümeler:")
        for cluster_id, cluster_info in clustering['clusters'].items():
            print(f"\n{cluster_id}:")
            print(f"Küme büyüklüğü: {cluster_info['size']}")
            print("İlk birkaç kitap:")
            for book in cluster_info['books'][:3]:
                print(f"- {book.get('Title', 'Başlık yok')}")