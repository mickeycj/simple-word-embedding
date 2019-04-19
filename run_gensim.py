import os
import sys
import re

from gensim.models import KeyedVectors, Word2Vec
from langdetect import detect
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import preprocessor as tweet

words = set(words.words())
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_links_and_hashtags(df):
    df['cleanText'] = df['text'].apply(lambda text: tweet.clean(str(text)))

    return df

def to_lowercase(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: text.lower())

    return df

def remove_contractions(df):
    def __remove_contractions(tweet):
        tweet = re.sub(r'’', '\'', tweet)
    
        tweet = re.sub(r'won\'t', 'will not', tweet)
        tweet = re.sub(r'can\'t', 'can not', tweet)
        
        tweet = re.sub(r'\'s', ' is', tweet)
        tweet = re.sub(r'\'m', ' am', tweet)
        tweet = re.sub(r'\'re', ' are', tweet)
        tweet = re.sub(r'\'ve', ' have', tweet)
        tweet = re.sub(r'\'ll', ' will', tweet)
        tweet = re.sub(r'\'d', ' would', tweet)
        tweet = re.sub(r'\'t', ' not', tweet)
        tweet = re.sub(r'n\'t', ' not', tweet)
        
        return tweet
    
    df['cleanText'] = df['cleanText'].apply(__remove_contractions)

    return df

def remove_punctuations(df):
    df['cleanText'] = df['cleanText'].str.replace(r'[^\w\s]', '')

    return df

def remove_whitespaces(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: str(text).strip())

    return df

def remove_non_english_tweets(df):
    df['lang'] = df['cleanText'].apply(lambda text: detect(text))
    df = df.drop(df[df['lang'] != 'en'].index)
    df = df.drop(columns=['lang'])

    return df

def lemmatize(df):
    def __get_pos(tag):
        tags = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

        try:
            return tags[tag]
        except KeyError:
            return wordnet.NOUN

    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(lemmatizer.lemmatize(word[0], pos=__get_pos(word[1][0]))
                                                                  for word in pos_tag(text.split())))

    return df

def remove_non_english_words(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(word for word in text.split()
                                                                  if word in words
                                                                  and len(word) > 1))

    return df

def remove_english_stopwords(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(word for word in text.split()
                                                                  if word not in stopwords))

    return df

def remove_empty_tweets(df):
    df = df.drop(df[df['cleanText'] == ''].index)

    return df

def preprocess(df):
    df = remove_links_and_hashtags(df)
    df = to_lowercase(df)
    df = remove_contractions(df)
    df = remove_punctuations(df)
    df = remove_whitespaces(df)
    df = remove_empty_tweets(df)
    df = remove_non_english_tweets(df)
    df = lemmatize(df)
    df = remove_non_english_words(df)
    df = remove_english_stopwords(df)
    df = remove_empty_tweets(df)

    return df

if __name__ == "__main__":
    command = sys.argv[1]
    if command == "train":
        print("Creating model...")
        w2v = Word2Vec(size=150, window=10, min_count=1, sg=1, workers=10)

        training_docs = []
        documents = pd.read_csv("./data/scraped_tweets.csv")[["text"]].dropna(subset=['text']).sample(frac=1).reset_index(drop=True)
        documents = preprocess(documents)
        num_documents = float(len(documents.index))
        print("Loading training documents: 0.00%...", end="\r")
        for index, row in documents.iterrows():
            print("Loading training documents: {:.2f}%...".format((index + 1) / num_documents * 100), end="\r")
            training_docs.append(row["cleanText"].split())
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
        documents = pd.read_csv("./data/scraped_tweets.csv", dtype=object)[["text"]].dropna(subset=['text']).sample(n=2).reset_index(drop=True)
        documents = preprocess(documents)
        testing_docs = list(map(lambda index__row: set(index__row[1]["cleanText"].split()), documents.iterrows()))

        if command == "test_word_sim":
            print("Finding similar words...")
            for document in testing_docs:
                for word in document:
                    print(word)
                    for word, sim in w2v.wv.most_similar_cosmul(positive=word, topn=3):
                        print(word, sim)
        else:
            print("Comparing two documents distance...")
            print(documents.iloc[0]["text"])
            print(documents.iloc[1]["text"])
            print(w2v.wmdistance(testing_docs[0], testing_docs[1]))
    else:
        print("\'{}\' command not found!".format(command))
