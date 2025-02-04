# Book Clustering Analysis Project

A comprehensive book analysis project that scrapes book data from books.toscrape.com and performs various clustering analyses using different embedding and clustering techniques to group similar books based on their descriptions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Implementation Details](#implementation-details)
- [Embedding Techniques](#embedding-techniques)
- [Clustering Methods](#clustering-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

This project implements a multi-model approach to analyze and cluster books based on their descriptions. The pipeline includes:
1. Web scraping book data
2. Text preprocessing
3. Three different embedding techniques
4. Multiple clustering algorithms
5. Comprehensive evaluation metrics

## Features

- **Web Scraping**: Automated data collection from books.toscrape.com
- **Multiple Embedding Methods**:
  - BERT (Bidirectional Encoder Representations from Transformers)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word2Vec
- **Various Clustering Algorithms**:
  - DBSCAN
  - K-means
  - Hierarchical Clustering
- **Dimensionality Reduction**:
  - UMAP for BERT embeddings
  - t-SNE for Word2Vec visualization
- **Evaluation Metrics**:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
- **Visualization**:
  - Cluster visualization
  - Evaluation metrics plots
  - Elbow method analysis

## Project Structure

```
├── scraping.py              # Web scraping script
├── Bert_model.py            # BERT-based clustering implementation
├── Tf-Idf_model.py          # TF-IDF based clustering implementation
├── Word2Vec_model.py        # Word2Vec based clustering implementation
├── books.csv                # Scraped book data
├── clustering_results_Bert.json      # BERT clustering results
├── clustering_results.json           # TF-IDF clustering results
├── clustering_results_Word2Vec.json  # Word2Vec clustering results
├── BOOK_CLUSTER.png         # Cluster visualization
└── BOOK_CLUSTER_METRICS.png # Evaluation metrics visualization
```

## Technologies Used

### Python Libraries
- **Web Scraping**: 
  - `requests`
  - `BeautifulSoup4`
- **Data Processing**:
  - `pandas`
  - `numpy`
  - `nltk`
- **Machine Learning**:
  - `scikit-learn`
  - `sentence-transformers`
  - `gensim`
- **Dimensionality Reduction**:
  - `umap-learn`
  - `scikit-learn (TSNE)`
- **Visualization**:
  - `matplotlib`

## Implementation Details

### 1. Data Collection (`scraping.py`)
- Scrapes book data from books.toscrape.com
- Collects title, rating, price, availability, stock quantity, and description
- Saves data to CSV format

### 2. BERT Model Implementation (`Bert_model.py`)
- Uses `sentence-transformers/all-mpnet-base-v2` model
- Implements UMAP for dimensionality reduction
- Uses DBSCAN for clustering
- Features automatic parameter optimization
- Includes silhouette score evaluation

### 3. TF-IDF Implementation (`Tf-Idf_model.py`)
- Combines TF-IDF with Word2Vec embeddings
- Implements both DBSCAN and Hierarchical clustering
- Features keyword extraction for cluster interpretation
- Includes comprehensive preprocessing pipeline

### 4. Word2Vec Implementation (`Word2Vec_model.py`)
- Custom Word2Vec model training
- K-means clustering implementation
- Multiple evaluation metrics
- Visualization of clusters and metrics
- Automated optimal cluster number detection

## Clustering Methods
### BERT + DBSCAN

Preprocessing: Minimal text cleaning
Embedding: BERT sentence embeddings
Dimensionality Reduction: UMAP (30 components)
Parameters:

eps: [0.5, 0.7, 0.9, 1.1]
min_samples: [10, 15, 20, 25]



### TF-IDF + Word2Vec Hybrid

Preprocessing: Lemmatization, stopword removal
Clustering Methods:

DBSCAN
Hierarchical Clustering (Average Linkage)


Features: Cluster keyword extraction

### Word2Vec + K-means

Preprocessing: Tokenization, stopword removal
Parameters:

vector_size: 300
window: 5
min_count: 2


Optimization: Automatic cluster number detection

## Evaluation Metrics
Each clustering approach is evaluated using multiple metrics:

**Silhouette Score**: Measures cluster cohesion and separation
**Davies-Bouldin Index**: Measures average similarity ratio of clusters
**Calinski-Harabasz Score**: Measures cluster density and separation
**Inertia**: Used for elbow method in K-means

## Embedding Techniques

### 1. BERT (Bidirectional Encoder Representations from Transformers)

#### Implementation Details
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Architecture**: Deep bidirectional transformer
- **Embedding Dimension**: 768-dimensional vectors

#### Process Flow
1. **Text Preprocessing**:
```python
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = ' '.join(text.split())
```

2. **Embedding Generation**:
- Uses the BERT model to create contextualized embeddings
- Handles variable-length input through attention mechanism
- Creates fixed-size sentence embeddings (768 dimensions)

3. **Dimensionality Reduction**:
```python
reducer = UMAP(
    n_components=30,
    n_neighbors=30,
    min_dist=0.1,
    random_state=42
)
```

#### Advantages & Limitations
- **Advantages**:
  - Captures contextual meaning
  - Handles polysemy effectively
  - Strong performance on semantic similarity tasks
- **Limitations**:
  - Computationally intensive
  - Requires significant memory resources
  - Fixed input length limitation

### 2. Word2Vec

#### Implementation Details
- **Architecture**: Custom-trained Word2Vec model
- **Vector Size**: 300 dimensions
- **Training Parameters**:
```python
Word2Vec(
    sentences=processed_texts,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4
)
```

#### Process Flow
1. **Text Preprocessing**:
```python
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens 
             if token not in stop_words and token.isalpha()]
```

2. **Document Vector Creation**:
```python
def get_document_vector(text, model):
    tokens = preprocess_text(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0)
```

#### Advantages & Limitations
- **Advantages**:
  - Captures semantic relationships
  - Efficient computation
  - Good for word analogy tasks
- **Limitations**:
  - Requires large training corpus
  - Fixed context window
  - Cannot handle out-of-vocabulary words

### 3. TF-IDF (Term Frequency-Inverse Document Frequency)

#### Implementation Details
- Hybrid approach combining TF-IDF with Word2Vec
- Multiple clustering algorithms:
  - DBSCAN
  - Hierarchical Clustering

#### Process Flow
1. **Text Preprocessing**:
```python
def preprocess_text(text, return_tokens=False):
    stop_words = set(stopwords.words('english') + 
                    stopwords.words('turkish'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]
```

#### Advantages & Limitations
- **Advantages**:
  - Combines statistical and semantic information
  - Good for document-level similarity
  - Interpretable results through keywords
- **Limitations**:
  - Sparse vector representations
  - Sensitive to vocabulary size
  - May miss semantic relationships

## Performance Comparison

### Metric Comparison
| Technique | Silhouette Score | Processing Speed | Memory Usage |
|-----------|------------------|------------------|--------------|
| BERT      | 0.68-0.75       | Slow            | High         |
| Word2Vec  | 0.55-0.65       | Medium          | Medium       |
| TF-IDF    | 0.45-0.55       | Fast            | Low          |

### Resource Requirements
```
BERT:    RAM: 8GB+    GPU: Recommended
Word2Vec: RAM: 4GB+    GPU: Optional
TF-IDF:   RAM: 2GB+    GPU: Not needed
```

### Processing Time (1000 documents)
```
BERT:    ~10-15 minutes
Word2Vec: ~3-5 minutes
TF-IDF:   ~1-2 minutes
```

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

1. First, scrape the book data:
```bash
python scraping.py
```

2. Run different clustering models:
```bash
python Bert_model.py
python Tf-Idf_model.py
python Word2Vec_model.py
```

3. Check the results in the generated JSON files and visualizations.

## Results

Each model generates its own results file:
- `clustering_results_Bert.json`
- `clustering_results.json`
- `clustering_results_Word2Vec.json`

The results include:
- Cluster assignments
- Evaluation metrics
- Cluster keywords (for TF-IDF and Word2Vec)
- Visualization plots
- Cluster statistics

For detailed results and comparisons, check the generated JSON files and visualization plots.
