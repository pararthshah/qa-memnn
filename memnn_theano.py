import numpy as np
import theano
import theano.tensor as T
import sys, random

from theano_util import *

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100):
        self.n_embedding = n_embedding
        self.lr = lr
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 2 * self.n_words
        self.n_embedding = n_embedding

        phi_x = T.vector('phi_x')
        phi_f1 = T.vector('phi_f1')
        phi_f1bar = T.vector('phi_f1bar')

        # Supporting memories
        phi_m0 = T.vector('phi_m0')

        # True word
        phi_r = T.vector('phi_r')

        # False words
        phi_rbar = T.vector('phi_rbar')

        self.U_O = init_shared_normal(self.n_embedding, self.n_D, 0.01)
        self.U_R = init_shared_normal(self.n_embedding, self.n_D, 0.01)

        cost = self.calc_cost(phi_x, phi_f1, phi_f1bar, phi_m0, phi_r, phi_rbar)
        params = [self.U_O, self.U_R]
        gradient = T.grad(cost, params)

        updates=[]
        for param, gparam in zip(params, gradient):
            updates.append((param, param - gparam * self.lr))

        self.train_function = theano.function(inputs = [phi_x, phi_f1, phi_f1bar, phi_m0, phi_r, phi_rbar],
                                         outputs = cost,
                                         updates = updates)

        phi_f = T.vector('phi_f')

        score_o = self.calc_score_o(phi_x, phi_f)
        self.predict_function_o = theano.function(inputs = [phi_x, phi_f], outputs = score_o)

        score_r = self.calc_score_r(phi_x, phi_f)
        self.predict_function_r = theano.function(inputs = [phi_x, phi_f], outputs = score_r)

    def calc_score(self, phi_x, phi_y, U):
        #return T.dot(T.dot(phi_x.T, self.U_O.T), T.dot(self.U_O, phi_y))
        return T.dot(U.dot(phi_x), U.dot(phi_y))

    def calc_score_o(self, phi_x, phi_y):
        return self.calc_score(phi_x, phi_y, self.U_O)

    def calc_score_r(self, phi_x, phi_y):
        return self.calc_score(phi_x, phi_y, self.U_R)

    def calc_cost(self, phi_x, phi_f1, phi_f1bar, phi_m0, phi_r, phi_rbar):
        correct_score1 = self.calc_score_o(phi_x, phi_f1)
        false_score1 = self.calc_score_o(phi_x, phi_f1bar)

        correct_score2 = self.calc_score_r(phi_x + phi_m0, phi_r)
        false_score2 = self.calc_score_r(phi_x + phi_m0, phi_rbar)

        cost = (
            T.maximum(0, self.margin - correct_score1 + false_score1) +
            T.maximum(0, self.margin - correct_score2 + false_score2)
        )
        return cost

    def train(self, dataset_bow, questions, num_words):
        for epoch in xrange(self.n_epochs):
            costs = []

            random.shuffle(questions)
            for i, question in enumerate(questions):
                article_no = question[0]
                line_no = question[1]
                question_phi = question[2]
                correct_stmt = question[4]
                seq = [i for i in range(line_no)]
                del seq[correct_stmt]
                false_stmt = random.choice(seq)
                #print article_no, line_no, correct_stmt, false_stmt
                phi_x = np.zeros((self.n_D,))
                phi_x[:num_words] = question_phi
                phi_f1 = np.zeros((self.n_D,))
                phi_f1[num_words:2*num_words] = dataset_bow[article_no][correct_stmt]
                phi_f1bar = np.zeros((self.n_D,))
                phi_f1bar[num_words:2*num_words] = dataset_bow[article_no][false_stmt]

                if article_no == 0 and line_no == 2:
                    corr_score = self.predict_function(phi_x, phi_f1)
                    fals_score = self.predict_function(phi_x, phi_f1bar)
                    print "[BEFORE] corr score: %f, false score: %f" % (corr_score, fals_score)

                cost = self.train_function(phi_x, phi_f1, phi_f1bar)
                costs.append(cost)

                if article_no == 0 and line_no == 2:
                    corr_score = self.predict_function(phi_x, phi_f1)
                    fals_score = self.predict_function(phi_x, phi_f1bar)
                    print "[ AFTER] corr score: %f, false score: %f" % (corr_score, fals_score)

            if epoch % 100 == 0:
                # print 'Epoch %i/%i' % (epoch + 1, self.n_epochs), np.mean(costs)
                sys.stdout.flush()

            # print np.mean(costs), np.mean(self.U_O.get_value()), np.max(self.U_O.get_value()), np.min(self.U_O.get_value())

    def predict(self, dataset, questions):
        correct_answers = 0
        wrong_answers = 0
        for i, question in enumerate(questions):
            article_no = question[0]
            line_no = question[1]
            question_phi = question[2]
            correct_stmt = question[4]

            phi_x = np.zeros((self.n_D,))
            phi_x[:num_words] = question_phi

            answer = -1
            max_score = -99999
            for i in range(line_no):
                phi_f = np.zeros((self.n_D,))
                phi_f[num_words:2*num_words] = dataset[article_no][i]

                #print phi_x, phi_f
                score = self.predict_function(phi_x, phi_f)
                if answer == -1 or score > max_score:
                    max_score = score
                    answer = i

            if answer == correct_stmt:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)

if __name__ == "__main__":
    training_dataset = sys.argv[1]
    test_dataset = training_dataset.replace('train', 'test')

    dataset, questions, word_to_id, num_words = parse_dataset(training_dataset)
    memNN = MemNN(n_words=num_words, n_embedding=100, lr=0.01, n_epochs=10, margin=1.0, word_to_id=word_to_id)
    memNN.train(dataset, questions)

    test_dataset, test_questions, _, _ = parse_dataset(test_dataset, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)
    memNN.predict(test_dataset, test_questions)
