import os
import sys
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words("english")

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    return " ".join(lemmatizer.lemmatize(w) for w in nltk.wordpunct_tokenize(text)
        if w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        print("Creating model...")
        w2v = Word2Vec(size=150, window=10, min_count=1, sg=1, workers=10)

        print("Loading training documents: 0.00%...", end="\r")
        training_docs = []
        documents = pd.read_csv("./data/scraped_tweets.csv")[["clean_text"]].dropna().sample(frac=1).reset_index(drop=True)
        num_documents = float(len(documents.index))
        for index, row in documents.iterrows():
            print("Loading training documents: {:.2f}%...".format((index + 1) / num_documents * 100), end="\r")
            training_docs.append(tokenize(preprocess(row["clean_text"])))
        print("Loading training documents: 100.00%...")

        print("Building vocabulary...")
        w2v.build_vocab(training_docs)

        print("Training...")
        w2v.train(sentences=training_docs, total_examples=len(training_docs), epochs=w2v.epochs)
        
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
    elif command == "test_word_sim" or command == "test_doc_sim":
        print("Loading model...")
        w2v = KeyedVectors.load("./kv/gensim_w2v.kv")

        print("Loading testing documents...")
        documents = pd.read_csv("./data/scraped_tweets.csv", dtype=object)[["text"]].dropna().sample(n=2).reset_index(drop=True)
        testing_docs = list(map(lambda index__row: set(tokenize(preprocess(index__row[1]["text"]))), documents.iterrows()))

        if command == "test_word_sim":
            print("Finding similar words...")
            for document in testing_docs:
                for word in document:
                    print(word)
                    for word, sim in w2v.wv.most_similar_cosmul(positive=word, topn=3):
                        print(word, sim)
        else:
            print("Comparing two documents distance...")
            print(documents.iloc[0]["question1"])
            print(documents.iloc[1]["question1"])
            print(w2v.wmdistance(testing_docs[0], testing_docs[1]))
    else:
        print("\'{}\' command not found!".format(command))
