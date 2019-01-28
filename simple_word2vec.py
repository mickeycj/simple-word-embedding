import numpy as np
import re
from collections import defaultdict

class Word2Vec():

    def __init__(self, settings):
        self.n = settings["n"]
        self.window_size = settings["window_size"]
        self.learning_rate = settings["learning_rate"]
        self.epochs = settings["epochs"]
    
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1

        return word_vec
    
    def generate_training_data(self, corpus):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        self.v_count = len(word_counts.keys())
        print("Unique words: {}".format(self.v_count))

        self.word_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.word_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.word_list))

        training_data = []
        for idx, sentence in enumerate(corpus):
            print("{}. {}".format(idx, sentence))
            sentence_length = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j >= 0 and j < sentence_length:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        
        return np.array(training_data)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum(axis=0)

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)

        return y_c, h, u

    def backprop(self, e, h, x):
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        dl_dw2 = np.outer(h, e)

        self.w1 = self.w1 - self.learning_rate * dl_dw1
        self.w2 = self.w2 - self.learning_rate * dl_dw2
    
    def train(self, training_data):
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))

        for i in range(0, self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_pass(w_t)
                
                e = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(e, h, w_t)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                self.loss += -2 * np.log(len(w_c)) - np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            
                print("EPOCH: {} | LOSS: {}".format(i + 1, self.loss))
    
    def word_sim(self, word, top_n):
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key=lambda word__sim: word__sim[1], reverse=True)
        
        return words_sorted[1:top_n+1]
