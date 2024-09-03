from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import LdaModel, Word2Vec
from gensim import corpora
import numpy as np


# Custom transformer for Word2Vec adapted to work under scikit-learn structure
class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim=25, window=16):
        self.w2v = None # In order to stock the w2v representation
        self.dim = dim  # Dimension of embeddings
        self.window = window # Dimension of the window

    def fit(self, X, y=None):
        sentences = [s.split() for s in X]
        self.w2v = Word2Vec(sentences, sg=1, vector_size=self.dim, window=self.window, min_count=1, workers=4) # We use by default Skip-grame
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.w2v.wv[w] for w in words if w in self.w2v.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in [s.split() for s in X]
        ])
    
class LDAEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics=6, passes=10):
        self.num_topics = num_topics
        self.passes = passes
        self.dictionary = None
        self.lda = None

    def fit(self, X, y=None):
        texts = [doc.split() for doc in X]
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda = LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=self.passes, random_state=42, minimum_probability=0.0)
        return self

    def transform(self, X):
        texts = [doc.split() for doc in X]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        # Ensuring consistent output by filling zero for absent topics
        matrix = np.zeros((len(texts), self.num_topics))
        for i, doc_bow in enumerate(corpus):
            for topic, prob in self.lda.get_document_topics(doc_bow, minimum_probability=0.0):
                matrix[i, topic] = prob
        return matrix
    


# Define different document vectorization techniques
vectorizers = [
    ("Bag-of-Words", CountVectorizer()),
    ("TF-IDF", TfidfVectorizer()),
    ("Word2Vec", MeanEmbeddingVectorizer()),
    ("LDA", LDAEmbeddingVectorizer())
]

# Define classification models
models = [
    ("SVC", SVC()),
    ("KNN", KNeighborsClassifier())
]
# Define hyperparameter grids for each model
param_grid_svc = {
    'model__C': np.linspace(0.1, 1, num=10),
    'model__gamma': ['scale'],
    'model__kernel': ['rbf']
}

param_grid_knn = {
    'model__n_neighbors': np.linspace(8, 15, num=7).astype(int)
}
# Create a list of model and vectorizer configurations
configurations = []
for model_name, model in models:
    if model_name == "SVC":
        param_grid = param_grid_svc
    elif model_name == "KNN":
        param_grid = param_grid_knn

    for vectorizer_name, vectorizer in vectorizers:
        configuration = {
            'model_name': model_name,
            'vectorizer_name': vectorizer_name,
            'model': model,
            'vectorizer': vectorizer,
            'param_grid': param_grid
        }
        configurations.append(configuration)

class Classifier():
    def __init__(self, column, configurations):
        self.column = column
        self.configurations = configurations
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results_cv = None
        self.y_pred = None

    def load_multiple_column(self, X_trainset_df, X_testset_df, y_trainset_df, y_testset_df):
        self.X_train = X_trainset_df[self.column].apply(' '.join)
        self.X_test = X_testset_df[self.column].apply(' '.join)
        self.y_train = y_trainset_df
        self.y_test = y_testset_df



