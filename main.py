import sys
from gensim.utils import simple_preprocess
import pandas as pd
from word2vec import Word2Vec

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    settings = {}
    settings["n"] = int(sys.argv[1])
    settings["window_size"] = int(sys.argv[2])
    settings["learning_rate"] = float(sys.argv[3])
    settings["epochs"] = int(sys.argv[4])

    print("Creating corpus...")
    corpus = []
    documents = pd.read_csv("./data/train.csv")[["question1", "question2", "is_duplicate"]].sample(n=50000).reset_index(drop=True)
    for index, row in documents.iterrows():
        print("Tokenize row {}".format(index))
        corpus.append(tokenize(row["question1"]))
        if row["is_duplicate"] == 0:
            corpus.append(tokenize(row["question2"]))

    w2v = Word2Vec(settings)    

    print("Creating training data...")
    training_data = w2v.generate_training_data(corpus)

    print("Training...")
    w2v.train(training_data)

    for word in ['the','quick','brown','fox','jumped','over','lazy','dog']:
        print(word)
        for word, sim in w2v.word_sim(word, 3):
            print(word, sim)
