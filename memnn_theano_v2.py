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

    def construct_phi(self, phi_type, bow=None, word_id=None):
        # type 0: question (phi_x)
        # type 1: supporting memory (phi_m*)
        # type 2: candidate memory (phi_y)
        # type 3: word vector
        assert(phi_type >= 0 and phi_type < 4)
        phi = np.zeros((3*self.n_words,))
        if phi_type < 3:
            assert(bow != None)
            phi[phi_type*self.n_words:(phi_type+1)*self.n_words] = bow
        else:
            assert(word_id != None and word_id < self.n_words)
            phi[2*self.n_words + word_id] = 1
        return phi

    def neg_sample(self, c, num):
        assert(c < num)
        assert(num > 1)
        f = random.randint(0, num-2)
        if f == c:
            f = num-1
        return f

    def find_m0(self, phi_x, statements, line_no, ignore=None):
        max_score = float("-inf")
        index_m0 = None
        m0 = None
        for i in xrange(line_no):
            if ignore and i == ignore:
                continue

            s = statements[i]
            phi_s = self.construct_phi(2, bow=s)

            score = self.predict_function_o(phi_x, phi_s)
            if score > max_score:
                max_score = score
                index_m0 = i
                m0 = s

        return index_m0, m0

    def train(self, dataset_bow, questions):
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

                # The question
                phi_x = self.construct_phi(0, bow=question_phi)
                # Correct statement 1
                phi_f1 = self.construct_phi(2, bow=dataset_bow[article_no][correct_stmt1])
                # Correct statement 2
                phi_f2 = self.construct_phi(2, bow=dataset_bow[article_no][correct_stmt2])
                # Correct word
                correct_word = question[3]
                phi_r = self.construct_phi(3, word_id=correct_word)

                # Find m0
                index_m0, m0 = self.find_m0(phi_x, dataset_bow[article_no], line_no)
                phi_m0 = self.construct_phi(1, bow=m0)

                # Find m1
                index_m1, m1 = self.find_m0(phi_x + phi_m0, dataset_bow[article_no], line_no, ignore=index_m0)
                phi_m1 = self.construct_phi(1, bow=m1)

                # False statement 1
                false_stmt1 = self.neg_sample(correct_stmt1, line_no)
                phi_f1bar = self.construct_phi(2, bow=dataset_bow[article_no][false_stmt1])

                # False statement 2
                false_stmt2 = self.neg_sample(correct_stmt2, line_no)
                phi_f2bar = self.construct_phi(2, bow=dataset_bow[article_no][false_stmt2])

                # False word
                false_word, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                if false_word == correct_word:
                    false_word = self.neg_sample(correct_word, self.n_words)
                phi_rbar = self.construct_phi(3, word_id=false_word)

                if article_no == 1 and line_no == 12:
                    print '[SAMPLE] %s\t%s' % (self.id_to_word[correct_word], self.id_to_word[false_word])
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
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
                    phi_m0 = self.construct_phi(1, bow=m0)
                    index_m1, m1 = self.find_m0(phi_x + phi_m0, dataset_bow[article_no], line_no, ignore=index_m0)
                    phi_m1 = self.construct_phi(1, bow=m1)
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                    print "[ AFTER] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d\ta:%s\ts:%.3f\tc:%s" % (
                        self.predict_function_o(phi_x, phi_f1),
                        self.predict_function_o(phi_x, phi_f1bar),
                        self.predict_function_o(phi_x + phi_m0, phi_f2),
                        self.predict_function_o(phi_x + phi_m0, phi_f2bar),
                        index_m0, index_m1,
                        self.id_to_word[w], score, self.id_to_word[correct_word]
                    )

            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def find_word(self, phi_x, verbose=False):
        max_score = float("-inf")
        best_word = -1
        for i in xrange(self.n_words):
            phi_r = self.construct_phi(3, word_id=i)
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

    def predict(self, dataset, questions):
        correct_answers = 0
        wrong_answers = 0
        fake_correct_answers = 0
        for i, question in enumerate(questions):
            article_no = question[0]
            line_no = question[1]
            question_phi = question[2]
            correct = question[3]

            phi_x = self.construct_phi(0, bow=question_phi)

            statements = dataset[article_no]

            phi_m0 = None
            phi_m1 = None
            if len(statements) == 0:
                print "Stupid question"
                continue
            elif len(statements) == 1:
                print "Stupid question?"
                phi_m0 = self.construct_phi(1, statements[0])
                phi_m1 = self.construct_phi(1, statements[0])
            else:
                index_m0, m0 = self.find_m0(phi_x, statements, line_no)
                phi_m0 = self.construct_phi(1, m0)
                index_m1, m1 = self.find_m0(phi_x + phi_m0, statements, line_no, ignore=index_m0)
                phi_m1 = self.construct_phi(1, m1)

                c1 = int(question[4].split(' ')[0])
                c2 = int(question[4].split(' ')[1])
                if index_m0 == c1 or index_m0 == c2 or index_m1 == c1 or index_m1 == c2:
                    fake_correct_answers += 1

            if article_no == 1:
                predicted, _ = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                print "%d, %d: predicted: %s, correct: %s" % (i, line_no, self.id_to_word[predicted], self.id_to_word[correct])
            else:
                predicted, _ = self.find_word(phi_x + phi_m0 + phi_m1)
            if predicted == correct:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong, %d fake_correct' % (correct_answers, wrong_answers, fake_correct_answers)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    train_dataset, train_questions, word_to_id, num_words = parse_dataset(train_file)
    test_dataset, test_questions, _, _ = parse_dataset(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)

    memNN = MemNN(n_words=num_words, n_embedding=100, lr=0.01, n_epochs=100, margin=1.0, word_to_id=word_to_id)
    memNN.train(train_dataset, train_questions)
    memNN.predict(test_dataset, test_questions)
