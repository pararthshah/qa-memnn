import numpy as np
import theano
import theano.tensor as T
import sys, random

from util import *

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100):
        self.n_embedding = n_embedding
        self.lr = lr
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 3 * self.n_words

        # Question
        phi_x = T.vector('phi_x')

        # True statements
        phi_f1 = T.vector('phi_f1')
        phi_f2 = T.vector('phi_f2')

        # False statements
        phi_f1bar = T.vector('phi_f1bar')
        phi_f2bar = T.vector('phi_f2bar')

        m0 = T.vector('m0')

        self.U_O = init_shared_normal(n_embedding, self.n_D, 0.01)
        cost = self.calc_cost(phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, m0)
        params = [self.U_O]
        gradient = T.grad(cost, params)

        updates=[]
        for param, gparam in zip(params, gradient):
            updates.append((param, param - gparam * self.lr))

        self.train_function = theano.function(inputs = [phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, m0],
                                         outputs = cost,
                                         updates = updates)

        # Candidate statement for prediction
        phi_f = T.vector('phi_f')

        score = self.calc_score(phi_x, phi_f)
        self.predict_function = theano.function(inputs = [phi_x, phi_f], outputs = score)

    def calc_score(self, phi_x, phi_y):
        return T.dot(self.U_O.dot(phi_x), self.U_O.dot(phi_y))

    def calc_cost(self, phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, m0):
        correct_score1 = self.calc_score(phi_x, phi_f1)
        false_score1 = self.calc_score(phi_x, phi_f1bar)

        correct_score2 = self.calc_score(phi_x + m0, phi_f2)
        false_score2 = self.calc_score(phi_x + m0, phi_f2bar)

        cost = (
            T.maximum(0, self.margin - correct_score1 + false_score1) +
            T.maximum(0, self.margin - correct_score2 + false_score2)
        )

        return cost

    def find_m0(self, phi_x, statements, line_no, ignore=None):
        max_score = float("-inf")
        m0 = None
        phi_m0 = None
        for i in xrange(line_no):
            if ignore and i == ignore:
                continue

            s = statements[i]
            phi_s = np.zeros((self.n_D,))
            phi_s[2*num_words:3*num_words] = s

            score = self.predict_function(phi_x, phi_s)
            if score > max_score:
                max_score = score
                m0 = i
                phi_m0 = phi_s

        return m0, phi_m0

    def train(self, dataset_bow, questions, num_words):
        for epoch in xrange(self.n_epochs):
            costs = []

            random.shuffle(questions)
            for i, question in enumerate(questions):
                article_no = question[0]
                line_no = question[1]
                question_phi = question[2]
                correct_stmts = question[4].split(' ')
                correct_stmt1 = int(correct_stmts[0])
                correct_stmt2 = int(correct_stmts[1])

                if line_no <= 2:
                    #print "Stupid question, article: %d, line: %d" % (article_no, line_no)
                    continue

                # False statement
                seq = [i for i in range(line_no)]
                del seq[correct_stmt1]
                false_stmt1 = random.choice(seq)

                del seq[correct_stmt2 if (correct_stmt1 > correct_stmt2) else (correct_stmt2 - 1)]
                false_stmt2 = random.choice(seq)

                # The question
                phi_x = np.zeros((self.n_D,))
                phi_x[:num_words] = question_phi

                # Correct statement 1
                phi_f1 = np.zeros((self.n_D,))
                phi_f1[num_words:2*num_words] = dataset_bow[article_no][correct_stmt1]

                # False statement 1
                phi_f1bar = np.zeros((self.n_D,))
                phi_f1bar[num_words:2*num_words] = dataset_bow[article_no][false_stmt1]

                # Find m0
                index_m0, phi_m0 = self.find_m0(phi_x, dataset_bow[article_no], line_no)

                # Correct statement 2
                phi_f2 = np.zeros((self.n_D,))
                phi_f2[num_words:2*num_words] = dataset_bow[article_no][correct_stmt2]

                # False statement 2
                phi_f2bar = np.zeros((self.n_D,))
                phi_f2bar[num_words:2*num_words] = dataset_bow[article_no][false_stmt2]

                if article_no == 1 and line_no == 12:
                    index_m1, _ = self.find_m0(phi_x + phi_m0, dataset_bow[article_no], line_no, ignore=index_m0)
                    print "[BEFORE] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d" % (
                        self.predict_function(phi_x, phi_f1),
                        self.predict_function(phi_x, phi_f1bar),
                        self.predict_function(phi_x + phi_m0, phi_f2),
                        self.predict_function(phi_x + phi_m0, phi_f2bar),
                        index_m0, index_m1
                    )

                cost = self.train_function(phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, phi_m0)
                costs.append(cost)

                if article_no == 1 and line_no == 12:
                    index_m0, _ = self.find_m0(phi_x, dataset_bow[article_no], line_no)
                    index_m1, _ = self.find_m0(phi_x + phi_m0, dataset_bow[article_no], line_no, ignore=index_m0)
                    print "[ AFTER] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d" % (
                        self.predict_function(phi_x, phi_f1),
                        self.predict_function(phi_x, phi_f1bar),
                        self.predict_function(phi_x + phi_m0, phi_f2),
                        self.predict_function(phi_x + phi_m0, phi_f2bar),
                        index_m0, index_m1
                    )

            print "Epoch %d: %f" % (epoch, np.mean(costs))

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

            answer = None
            max_score = float("-inf")

            statements = dataset[article_no]

            index_m0 = None
            index_m1 = None
            if len(statements) <= 1:
                index_m0 = len(statements) - 1
                index_m1 = len(statements) - 1
            else:
                index_m0, phi_m0 = self.find_m0(phi_x, statements, line_no)
                index_m1, _ = self.find_m0(phi_x + phi_m0, statements, line_no, ignore=index_m0)

            #answer = str(index_m0) + ' ' + str(index_m1)
            #if answer == correct_stmt:
            #    correct_answers += 1
            #else:
            #    wrong_answers += 1
            c1 = int(correct_stmt.split(' ')[0])
            c2 = int(correct_stmt.split(' ')[1])
            if index_m0 == c1 or index_m0 == c2 or index_m1 == c1 or index_m1 == c2:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)

if __name__ == "__main__":
    training_dataset = sys.argv[1]
    test_dataset = training_dataset.replace('train', 'test')

    dataset, questions, word_to_id, num_words = parse_dataset(training_dataset)
    dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), dataset)
    questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), questions)
    memNN = MemNN(n_words=num_words, n_embedding=100, lr=0.01, n_epochs=10, margin=1.0)
    memNN.train(dataset_bow, questions_bow, num_words)

    test_dataset, test_questions, _, _ = parse_dataset(test_dataset)
    test_dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), test_dataset)
    test_questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), test_questions)
    memNN.predict(test_dataset_bow, test_questions_bow)
