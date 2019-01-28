import os
import sys
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        print("Creating corpus...")
        corpus = []
        documents = pd.read_csv("./data/train.csv")[["question1", "question2", "is_duplicate"]].sample(frac=1).reset_index(drop=True)
        for index, row in documents.iterrows():
            corpus.append(tokenize(row["question1"]))
            if row["is_duplicate"] == 0:
                corpus.append(tokenize(row["question2"]))
        
        w2v = Word2Vec(size=150, window=10, min_count=2, sg=1, workers=10)

        print("Creating training data...")
        w2v.build_vocab(corpus)

        print("Training...")
        w2v.train(sentences=corpus, total_examples=len(corpus), epochs=w2v.epochs)
        
        print("Creating model...")
        model_path = "./models/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        w2v.save("{}{}".format(model_path, "gensim_w2v.model"))

        print("Training finished!")
    elif command == "test":
        print("Creating test words...")
        documents = pd.read_csv("./data/test.csv")[["question1"]].sample(n=3).reset_index(drop=True)
        documents = map(lambda index__row: tokenize(index__row[1]["question1"]), documents)

        print("Loading model...")
        w2v = Word2Vec.load("./models/gensim_w2v.model")

        print("Finding similar words...")
        for document in documents:
            for word in document:
                print(word)
                for word, sim in w2v.wv.most_similar(positive=word, topn=5):
                    print(word, sim)
    else:
        print("\'{}\' command not found!".format(command))
