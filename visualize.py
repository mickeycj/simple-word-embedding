import sys

from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words("english")

lemmatizer = WordNetLemmatizer()

def lemmatize(word):
    noun, verb = lemmatizer.lemmatize(word), lemmatizer.lemmatize(word, "v")
    if len(word) == len(noun):
        return verb
    else:
        return noun

def preprocess(text):
    return " ".join(lemmatize(w) for w in nltk.wordpunct_tokenize(text)
        if w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    dimension = 3 if len(sys.argv) > 1 and sys.argv[1] == "3D" else 2

    print("Loading pre-trained word-vectors...")
    w2v = KeyedVectors.load("./kv/gensim_w2v.kv")

    print("Loading documents for visualization...")
    documents = pd.read_csv("./data/train.csv", dtype=object)[["question1"]].dropna().sample(n=117).reset_index(drop=True)
    last_indices = []
    words = []
    for _, row in documents.iterrows():
        tokens = tokenize(preprocess(row["question1"]))
        if len(last_indices) > 0:
            last_indices.append(last_indices[len(last_indices) - 1] + len(tokens))
        else:
            last_indices.append(len(tokens))
        words.extend(tokens)
    
    print("Creating vectors...")
    vectors = list(map(lambda word: w2v[word], words))

    print("Creating document vectors...")
    document_vectors = []
    first_index = 0
    (vector_size,) = w2v[words[0]].shape
    for last_index in last_indices:
        vector = []
        for i in range(vector_size):
            v_list = [v[i] for v in vectors[first_index:last_index]]
            if len(v_list) > 0:
                v = sum(v_list) / len(v_list)
                vector.append(v)
        if len(vector) > 0:
            document_vectors.append(vector)
        first_index = last_index
    document_vectors = list(document_vectors)

    print("Performing PCA...")
    document_vectors = StandardScaler().fit_transform(document_vectors)
    document_vectors = PCA(0.85).fit_transform(document_vectors)

    print("Performing t-SNE...")
    document_vectors = TSNE(n_components=dimension).fit_transform(document_vectors)

    print("Creating document clusters...")
    clustering = KMeans().fit(document_vectors)
    centers = clustering.cluster_centers_
    labels = clustering.labels_
    clusters = {}
    for doc_coordinate, label in zip(document_vectors, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(doc_coordinate)
    
    if dimension == 3:
        print("Visualizing document clusters in 3D...")
        fig = plt.gcf()
        fig.canvas.set_window_title("Document Clusters (3D)")
        ax = fig.add_subplot(111, projection="3d")
        for label in clusters.keys():
            x_list = [doc_coordinate[0] for doc_coordinate in clusters[label]]
            y_list = [doc_coordinate[1] for doc_coordinate in clusters[label]]
            z_list = [doc_coordinate[2] for doc_coordinate in clusters[label]]
            x_center = sum(x_list) / len(x_list)
            y_center = sum(y_list) / len(y_list)
            z_center = sum(z_list) / len(z_list)
            color = next(ax._get_lines.prop_cycler)["color"]
            ax.scatter(x_center, y_center, z_center, marker="o", s=75, color=color)
            ax.scatter(x_list, y_list, z_list, marker="o", s=15, color=color)
        plt.show()
    else:
        print("Visualizing document clusters in 2D...")
        fig = plt.gcf()
        fig.canvas.set_window_title("Document Clusters (2D)")
        ax = fig.add_subplot(111)
        for label in clusters.keys():
            x_list = [doc_coordinate[0] for doc_coordinate in clusters[label]]
            y_list = [doc_coordinate[1] for doc_coordinate in clusters[label]]
            x_center = sum(x_list) / len(x_list)
            y_center = sum(y_list) / len(y_list)
            color = next(ax._get_lines.prop_cycler)["color"]
            ax.scatter(x_center, y_center, marker="o", s=75, color=color)
            ax.scatter(x_list, y_list, marker="o", s=15, color=color)
        plt.show()
