import sys
from gensim.utils import simple_preprocess
import pandas as pd

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    settings = {}
    settings["n"] = int(sys.argv[1])
    settings["window_size"] = int(sys.argv[2])

    print("Creating corpus...")
    corpus = []
    documents = pd.read_csv("./data/train.csv")[["question1", "question2", "is_duplicate"]].sample(frac=1).reset_index(drop=True)
    for index, row in documents.iterrows():
        corpus.append(tokenize(row["question1"]))
        if row["is_duplicate"] == 0:
            corpus.append(tokenize(row["question2"]))
    
    from gensim.models import Word2Vec   
    w2v = Word2Vec(size=settings["n"], window=settings["window_size"], min_count=2, sg=1, workers=10)

    print("Creating training data...")
    w2v.build_vocab(corpus)

    print("Training...")
    w2v.train(sentences=corpus, total_examples=len(corpus), epochs=w2v.epochs)

    for word in ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'lazy', 'dog']:
        print(word)
        for word, sim in w2v.wv.most_similar(positive=word, topn=5):
            print(word, sim)
