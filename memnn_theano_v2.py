import numpy as np
import theano
import theano.tensor as T
import sys, random, pprint

from util import *

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100, word_to_id=None):
        self.n_embedding = n_embedding
        self.lr = lr
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 3 * self.n_words

        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

        # Question
        phi_x = T.vector('phi_x')

        # True statements
        phi_f1 = T.vector('phi_f1')
        phi_f2 = T.vector('phi_f2')

        # False statements
        phi_f1bar = T.vector('phi_f1bar')
        phi_f2bar = T.vector('phi_f2bar')

        # Supporting memories
        phi_m0 = T.vector('phi_m0')
        phi_m1 = T.vector('phi_m1')

        # True word
        phi_r = T.vector('phi_r')

        # False words
        phi_rbar = T.vector('phi_rbar')

        self.U_O = init_shared_normal(n_embedding, self.n_D, 0.01)
        self.U_R = init_shared_normal(n_embedding, self.n_D, 0.01)

        cost = self.calc_cost(phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, phi_m0, phi_m1, phi_r, phi_rbar)
        params = [self.U_O, self.U_R]
        gradient = T.grad(cost, params)

        updates=[]
        for param, gparam in zip(params, gradient):
            updates.append((param, param - gparam * self.lr))

        self.train_function = theano.function(
            inputs = [phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, phi_m0, phi_m1, phi_r, phi_rbar],
            outputs = cost,
            updates = updates)

        # Candidate statement for prediction
        phi_f = T.vector('phi_f')

        score_o = self.calc_score_o(phi_x, phi_f)
        self.predict_function_o = theano.function(inputs = [phi_x, phi_f], outputs = score_o)

        score_r = self.calc_score_r(phi_x, phi_f)
        self.predict_function_r = theano.function(inputs = [phi_x, phi_f], outputs = score_r)

    def calc_score(self, phi_x, phi_y, U):
        return T.dot(U.dot(phi_x), U.dot(phi_y))

    def calc_score_o(self, phi_x, phi_y):
        return self.calc_score(phi_x, phi_y, self.U_O)

    def calc_score_r(self, phi_x, phi_y):
        return self.calc_score(phi_x, phi_y, self.U_R)

    def calc_cost(self, phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, phi_m0, phi_m1, phi_r, phi_rbar):
        correct_score1 = self.calc_score_o(phi_x, phi_f1)
        false_score1 = self.calc_score_o(phi_x, phi_f1bar)

        correct_score2 = self.calc_score_o(phi_x + phi_m0, phi_f2)
        false_score2 = self.calc_score_o(phi_x + phi_m0, phi_f2bar)

        correct_score3 = self.calc_score_r(phi_x + phi_m0 + phi_m1, phi_r)
        false_score3 = self.calc_score_r(phi_x + phi_m0 + phi_m1, phi_rbar)

        cost = (
            T.maximum(0, self.margin - correct_score1 + false_score1) +
            T.maximum(0, self.margin - correct_score2 + false_score2) +
            T.maximum(0, self.margin - correct_score3 + false_score3)
        )

        return cost

    def find_m0(self, phi_x, statements, line_no, ignore=None):
        max_score = float("-inf")
        index_m0 = None
        m0 = None
        for i in xrange(line_no):
            if ignore and i == ignore:
                continue

            s = statements[i]
            phi_s = np.zeros((self.n_D,))
            phi_s[num_words:2*num_words] = s

            score = self.predict_function_o(phi_x, phi_s)
            if score > max_score:
                max_score = score
                index_m0 = i
                m0 = s

        return index_m0, m0

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

                if line_no <= 1:
                    continue

                correct_word = question[3]
                phi_r = np.zeros((self.n_D,))
                phi_r[num_words + correct_word] = 1.0

                def neg_sample(c, num):
                    f = c
                    while f == c:
                        f = random.randint(0, num-1)
                    return f

                false_word = neg_sample(correct_word, num_words)
                phi_rbar = np.zeros((self.n_D,))
                phi_rbar[num_words + false_word] = 1.0

                # False statement
                false_stmt1 = neg_sample(correct_stmt1, line_no)
                false_stmt2 = neg_sample(correct_stmt2, line_no)

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
                index_m0, m0 = self.find_m0(phi_x, dataset_bow[article_no], line_no)
                phi_m0 = np.zeros((self.n_D,))
                phi_m0[2*num_words:3*num_words] = m0

                # Correct statement 2
                phi_f2 = np.zeros((self.n_D,))
                phi_f2[num_words:2*num_words] = dataset_bow[article_no][correct_stmt2]

                # False statement 2
                phi_f2bar = np.zeros((self.n_D,))
                phi_f2bar[num_words:2*num_words] = dataset_bow[article_no][false_stmt2]

                # Find m1
                index_m1, m1 = self.find_m0(phi_x, dataset_bow[article_no], line_no, ignore=index_m0)
                phi_m1 = np.zeros((self.n_D,))
                phi_m1[2*num_words:3*num_words] = m1

                if article_no == 1 and line_no == 12:
                    print '[SAMPLE] %s\t%s' % (self.id_to_word[correct_word], self.id_to_word[false_word])
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, num_words, verbose=False)
                    print "[BEFORE] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d\ta:%s\ts:%.3f\tc:%s" % (
                        self.predict_function_o(phi_x, phi_f1),
                        self.predict_function_o(phi_x, phi_f1bar),
                        self.predict_function_o(phi_x + phi_m0, phi_f2),
                        self.predict_function_o(phi_x + phi_m0, phi_f2bar),
                        index_m0, index_m1,
                        self.id_to_word[w], score, self.id_to_word[correct_word]
                    )

                cost = self.train_function(phi_x, phi_f1, phi_f1bar, phi_f2, phi_f2bar, phi_m0, phi_f1, phi_r, phi_rbar)
                costs.append(cost)

                if article_no == 1 and line_no == 12:
                    index_m0, m0 = self.find_m0(phi_x, dataset_bow[article_no], line_no)
                    phi_m0 = np.zeros((self.n_D,))
                    phi_m0[2*num_words:3*num_words] = m0
                    index_m1, m1 = self.find_m0(phi_x + phi_m0, dataset_bow[article_no], line_no, ignore=index_m0)
                    phi_m1 = np.zeros((self.n_D,))
                    phi_m1[2*num_words:3*num_words] = m1
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, num_words, verbose=False)
                    print "[ AFTER] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d\ta:%s\ts:%.3f\tc:%s" % (
                        self.predict_function_o(phi_x, phi_f1),
                        self.predict_function_o(phi_x, phi_f1bar),
                        self.predict_function_o(phi_x + phi_m0, phi_f2),
                        self.predict_function_o(phi_x + phi_m0, phi_f2bar),
                        index_m0, index_m1,
                        self.id_to_word[w], score, self.id_to_word[correct_word]
                    )

            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def find_word(self, phi_x, num_words, verbose=False):
        max_score = float("-inf")
        best_word = -1
        for i in xrange(num_words):
            phi_r = np.zeros((self.n_D,))
            phi_r[num_words + i] = 1
            score = self.predict_function_r(phi_x, phi_r)
            if verbose:
                print '[  FIND] w:%s\ts:%.3f' % (
                    self.id_to_word[i],
                    score
                )
            if score > max_score:
                max_score = score
                best_word = i

        assert(best_word >= 0)
        return best_word, score

    def predict(self, dataset, questions, num_words):
        correct_answers = 0
        wrong_answers = 0
        fake_correct_answers = 0
        for i, question in enumerate(questions):
            article_no = question[0]
            line_no = question[1]
            question_phi = question[2]
            correct = question[3]

            phi_x = np.zeros((self.n_D,))
            phi_x[:num_words] = question_phi

            statements = dataset[article_no]

            phi_m0 = np.zeros((self.n_D,))
            phi_m1 = np.zeros((self.n_D,))
            if len(statements) == 0:
                print "Stupid question"
                continue
            elif len(statements) == 1:
                print "Stupid question?"
                phi_m0[2*num_words:3*num_words] = statements[0]
                phi_m1[2*num_words:3*num_words] = statements[0]
            else:
                index_m0, m0 = self.find_m0(phi_x, statements, line_no)
                phi_m0[2*num_words:3*num_words] = m0
                index_m1, m1 = self.find_m0(phi_x + phi_m0, statements, line_no, ignore=index_m0)
                phi_m1[2*num_words:3*num_words] = m1

                c1 = int(question[4].split(' ')[0])
                c2 = int(question[4].split(' ')[1])
                if index_m0 == c1 or index_m0 == c2 or index_m1 == c1 or index_m1 == c2:
                    fake_correct_answers += 1


            predicted, _ = self.find_word(phi_x + phi_m0 + phi_m1, num_words)

            #print "predicted: %s, correct: %s" % (self.id_to_word[predicted], self.id_to_word[correct])
            if predicted == correct:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong, %d fake_correct' % (correct_answers, wrong_answers, fake_correct_answers)

if __name__ == "__main__":
    training_dataset = sys.argv[1]
    test_dataset = training_dataset.replace('train', 'test')

    dataset, questions, word_to_id, num_words = parse_dataset(training_dataset)
    dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), dataset)
    questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), questions)
    memNN = MemNN(n_words=num_words, n_embedding=20, lr=0.01, n_epochs=100, margin=1.0, word_to_id=word_to_id)
    memNN.train(dataset_bow, questions_bow, num_words)

    test_dataset, test_questions, _, _ = parse_dataset(test_dataset, word_id=num_words, word_to_id=word_to_id)
    test_dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), test_dataset)
    test_questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), test_questions)
    memNN.predict(test_dataset_bow, test_questions_bow, num_words)
