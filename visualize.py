from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words("english")

def preprocess(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text)
        if w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
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
    scaled_vectors = StandardScaler().fit_transform(document_vectors)
    pca_results = PCA(0.85).fit_transform(scaled_vectors)

    print("Performing t-SNE...")
    tsne_results = TSNE(n_components=2).fit_transform(pca_results)
    
    print("Visualizing documents...")
    fig = plt.gcf()
    fig.canvas.set_window_title("Document Representations")
    first_index = 0
    times = 0
    for index, doc_coordinate in enumerate(tsne_results):
        plt.scatter(doc_coordinate[0], doc_coordinate[1], marker="o", label="Doc. {}".format(index + 1))
        first_index = last_index
    if len(sys.argv) > 1 and sys.argv[1] == "show_legend":
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    plt.show()
