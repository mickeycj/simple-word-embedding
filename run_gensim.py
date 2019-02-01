import os
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
import pandas as pd

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words("english")

def preprocess(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text)
        if w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        print("Creating corpus...")
        corpus = []
        documents = pd.read_csv("./data/train.csv")[["question1", "question2", "is_duplicate"]].dropna().sample(frac=1).reset_index(drop=True)
        for index, row in documents.iterrows():
            corpus.append(tokenize(preprocess(row["question1"])))
            if row["is_duplicate"] == 0:
                corpus.append(tokenize(preprocess(row["question2"])))
        
        print("Creating model...")
        w2v = Word2Vec(size=150, window=10, min_count=2, sg=1, workers=10)

        print("Creating training data...")
        w2v.build_vocab(corpus)

        print("Training...")
        w2v.train(sentences=corpus, total_examples=len(corpus), epochs=w2v.epochs)
        
        print("Saving model...")
        model_path = "./models/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        w2v.save("{}{}".format(model_path, "gensim_w2v.model"))
        kv_path = "./kv/"
        if not os.path.exists(kv_path):
            os.mkdir(kv_path)
        w2v.wv.save("{}{}".format(kv_path, "gensim_w2v.kv"))

        print("Training finished!")
    elif command == "test":
        print("Creating test words...")
        documents = pd.read_csv("./data/test.csv", dtype=object)[["question1"]].dropna().sample(n=3).reset_index(drop=True)
        documents = map(lambda index__row: tokenize(preprocess(index__row[1]["question1"])), documents.iterrows())

        print("Loading model...")
        w2v = KeyedVectors.load("./kv/gensim_w2v.kv")

        print("Finding similar words...")
        for document in documents:
            for word in document:
                print(word)
                for word, sim in w2v.wv.most_similar(positive=word, topn=3):
                    print(word, sim)
    else:
        print("\'{}\' command not found!".format(command))
