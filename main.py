import sys
from word2vec import Word2Vec

if __name__ == "__main__":
    settings = {}
    settings["n"] = int(sys.argv[1])
    settings["epochs"] = int(sys.argv[2])
    settings["window_size"] = int(sys.argv[3])
    settings["learning_rate"] = float(sys.argv[4])

    corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]

    w2v = Word2Vec(settings)    

    training_data = w2v.generate_training_data(corpus)

    w2v.train(training_data)

    for word in ['the','quick','brown','fox','jumped','over','lazy','dog']:
        print(word)
        for word, sim in w2v.word_sim(word, 3):
            print(word, sim)
