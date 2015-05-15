import numpy as np
import theano
import theano.tensor as T
import sys, random, pprint

from theano_util import *

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100, momentum=0.9, word_to_id=None):
        self.n_embedding = n_embedding
        self.lr = lr
        self.momentum = momentum
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 3 * self.n_words + 3

        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

        # Question
        phi_x = T.vector('phi_x')

        # True statements
        phi_f1_1 = T.vector('phi_f1_1')
        phi_f2_1 = T.vector('phi_f2_1')

        # False statements
        phi_f1_2 = T.vector('phi_f1_2')
        phi_f2_2 = T.vector('phi_f2_2')

        # Supporting memories
        phi_m0 = T.vector('phi_m0')
        phi_m1 = T.vector('phi_m1')

        # True word
        phi_r = T.vector('phi_r')

        # False words
        phi_rbars = T.matrix('phi_rbars')

        self.U_O = init_shared_normal(n_embedding, self.n_D, 0.01)
        self.U_R = init_shared_normal(n_embedding, self.n_D, 0.01)

        # Total S_R cost for all sampled words
        tot_sr_cost = T.scalar('sr_cost')

        cost = self.calc_cost(phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, phi_m0, phi_m1, phi_r, phi_rbars, tot_sr_cost)
        params = [self.U_O, self.U_R]
        gradient = T.grad(cost, params)

        l_rate = T.scalar('l_rate')

        updates=[]
        for param, gparam in zip(params, gradient):
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - param_update * l_rate))
            updates.append((param_update, self.momentum*param_update + (1. - self.momentum)*gparam))

        self.train_function = theano.function(
            inputs = [phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, \
                      phi_m0, phi_m1, phi_r, phi_rbars, \
                      theano.Param(l_rate, default=self.lr), \
                      theano.Param(tot_sr_cost, default=0.0)],
            outputs = cost,
            updates = updates)

        # Candidate statement for prediction
        phi_f = T.vector('phi_f')

        score_o = self.calc_score_o(phi_x, phi_f)
        self.predict_function_o = theano.function(inputs = [phi_x, phi_f], outputs = score_o)

        score_r = self.calc_score_r(phi_x, phi_f)
        self.predict_function_r = theano.function(inputs = [phi_x, phi_f], outputs = score_r)

    def calc_score_o(self, phi_x, phi_y_yp_t):
        return T.dot(self.U_O.dot(phi_x), self.U_O.dot(phi_y_yp_t))

    def calc_score_r(self, phi_x, phi_y):
        return T.dot(self.U_R.dot(phi_x), self.U_R.dot(phi_y))

    # phi_f1_1 = phi_f1 - phi_f1bar + phi_t1_1
    # phi_f1_2 = phi_f1bar - phi_f1 + phi_t1_2
    def calc_cost(self, phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, phi_m0, phi_m1, phi_r, phi_rbars, tot_sr_cost):
        score1_1 = self.calc_score_o(phi_x, phi_f1_1)
        score1_2 = self.calc_score_o(phi_x, phi_f1_2)

        score2_1 = self.calc_score_o(phi_x + phi_m0, phi_f2_1)
        score2_2 = self.calc_score_o(phi_x + phi_m0, phi_f2_2)

        s_o_cost = (
            T.maximum(0, self.margin - score1_1) + T.maximum(0, self.margin + score1_2) +
            T.maximum(0, self.margin - score2_1) + T.maximum(0, self.margin + score2_2)
        )

        def compute_sr_cost(phi_rbar, correct_score):
            false_score = self.calc_score_r(phi_x + phi_m0 + phi_m1, phi_rbar)
            return T.maximum(0, self.margin - correct_score + false_score)

        correct_score3 = self.calc_score_r(phi_x + phi_m0 + phi_m1, phi_r)
        sr_costs, sr_updates = theano.reduce(lambda phi_rbar, tot_sr_cost: tot_sr_cost + compute_sr_cost(phi_rbar, correct_score3),
                                             sequences=phi_rbars, outputs_info=[{'initial': tot_sr_cost}])

        cost = s_o_cost + sr_costs
        return cost

    def construct_phi(self, phi_type, bow=None, word_id=None, ids=None):
        # type 0: question (phi_x)
        # type 1: supporting memory (phi_m*)
        # type 2: candidate memory (phi_y)
        # type 3: word vector
        # type 4: write-time features
        assert(phi_type >= 0 and phi_type < 5)
        phi = np.zeros((3*self.n_words + 3,))
        if phi_type < 3:
            assert(bow is not None)
            phi[phi_type*self.n_words:(phi_type+1)*self.n_words] = bow
        elif phi_type == 3:
            assert(word_id != None and word_id < self.n_words)
            phi[2*self.n_words + word_id] = 1
        else:
            assert(ids != None and len(ids) == 3)
            if ids[0] > ids[1]: phi[3*self.n_words] = 1
            if ids[0] > ids[2]: phi[3*self.n_words+1] = 1
            if ids[1] > ids[2]: phi[3*self.n_words+2] = 1
        return phi

    # returns (phi_y - phi_yp + phi_t)
    def construct_wt_phi(self, index_x, index_y, index_yp, y, yp):
        phi_y = self.construct_phi(2, bow=y)
        phi_yp = self.construct_phi(2, bow=yp)
        phi_t = self.construct_phi(4, ids=[index_x, index_y, index_yp])
        return phi_y - phi_yp + phi_t

    def neg_sample(self, c, num):
        assert(c < num)
        assert(num > 1)
        f = random.randint(0, num-2)
        if f == c:
            f = num-1
        return f

    def find_m0(self, index_x, phi_x, statements, ignore=None):
        max_score = float("-inf")
        index_m0 = 0
        m0 = statements[0]
        for i in xrange(1,len(statements)):
            if ignore and i == ignore:
                continue

            s = statements[i]
            phi_s = self.construct_wt_phi(index_x, i, index_m0, s, m0)

            if self.predict_function_o(phi_x, phi_s) >= 0:
                index_m0 = i
                m0 = s

        return index_m0, m0

    def train(self, dataset_bow, questions, lr_schedule=None):
        l_rate = self.lr
        for epoch in xrange(self.n_epochs):
            costs = []

            if lr_schedule != None and epoch in lr_schedule:
                l_rate = lr_schedule[epoch]

            random.shuffle(questions)
            for i, question in enumerate(questions):
                article_no = question[0]
                article = dataset_bow[article_no]
                line_no = question[1]
                question_phi = question[2]
                correct_stmts = question[4].split(' ')
                correct_stmt1 = int(correct_stmts[0])
                correct_stmt2 = int(correct_stmts[1])

                if line_no <= 1:
                    continue

                # The question
                phi_x = self.construct_phi(0, bow=question_phi)

                # Find m0
                index_m0, m0 = self.find_m0(line_no, phi_x, article[:line_no])
                phi_m0 = self.construct_phi(1, bow=m0)

                # Find m1
                index_m1, m1 = self.find_m0(index_m0, phi_x + phi_m0, article[:line_no], ignore=index_m0)
                phi_m1 = self.construct_phi(1, bow=m1)

                # False statement 1
                false_stmt1 = index_m0
                if false_stmt1 == correct_stmt1:
                    false_stmt1 = self.neg_sample(correct_stmt1, line_no)
                phi_f1_1 = self.construct_wt_phi(line_no, correct_stmt1, false_stmt1, article[correct_stmt1], article[false_stmt1])
                phi_f1_2 = self.construct_wt_phi(line_no, false_stmt1, correct_stmt1, article[false_stmt1], article[correct_stmt1])

                # False statement 2
                false_stmt2 = index_m1
                if false_stmt2 == correct_stmt2:
                    false_stmt2 = self.neg_sample(correct_stmt2, line_no)
                phi_f2_1 = self.construct_wt_phi(line_no, correct_stmt2, false_stmt2, article[correct_stmt2], article[false_stmt2])
                phi_f2_2 = self.construct_wt_phi(line_no, false_stmt2, correct_stmt2, article[false_stmt2], article[correct_stmt2])

                # Correct word
                correct_word = question[3]
                phi_r = self.construct_phi(3, word_id=correct_word)

                # False word
                false_word_ids = [i for i in range(self.n_words)]
                del false_word_ids[correct_word]
                # Find the highest ranking word, if it isnt the correct word, add it to list
                # Possible that this word will be added twice, but that is okay
                false_word1, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                if false_word1 != correct_word:
                    false_word_ids.insert(0, false_word1)
                # Clip no. of samples to 20
                false_word_ids = false_word_ids[:min(20,len(false_word_ids))]
                phi_rbars = np.vstack(tuple(map(lambda word_id: self.construct_phi(3, word_id=word_id), false_word_ids)))

                if article_no == 1 and line_no == 12:
                    print '[SAMPLE] %s\t%s' % (self.id_to_word[correct_word], self.id_to_word[false_word1])
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                    print "[BEFORE] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d\ta:%s\ts:%.3f\tc:%s" % (
                        self.predict_function_o(phi_x, phi_f1_1),
                        self.predict_function_o(phi_x, phi_f1_2),
                        self.predict_function_o(phi_x + phi_m0, phi_f2_1),
                        self.predict_function_o(phi_x + phi_m0, phi_f2_2),
                        index_m0, index_m1,
                        self.id_to_word[w], score, self.id_to_word[correct_word]
                    )

                cost = self.train_function(phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, \
                                           phi_m0, phi_m1, phi_r, phi_rbars, \
                                           l_rate)
                costs.append(cost)

                if article_no == 1 and line_no == 12:
                    index_m0, m0 = self.find_m0(line_no, phi_x, article[:line_no])
                    phi_m0 = self.construct_phi(1, bow=m0)
                    index_m1, m1 = self.find_m0(index_m0, phi_x + phi_m0, article[:line_no], ignore=index_m0)
                    phi_m1 = self.construct_phi(1, bow=m1)
                    w, score = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                    print "[ AFTER] %.3f\t%.3f\t%.3f\t%.3f\tm0:%d\tm1:%d\ta:%s\ts:%.3f\tc:%s" % (
                        self.predict_function_o(phi_x, phi_f1_1),
                        self.predict_function_o(phi_x, phi_f1_2),
                        self.predict_function_o(phi_x + phi_m0, phi_f2_1),
                        self.predict_function_o(phi_x + phi_m0, phi_f2_2),
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
                index_m0, m0 = self.find_m0(line_no, phi_x, statements[:line_no])
                phi_m0 = self.construct_phi(1, m0)
                index_m1, m1 = self.find_m0(index_m0, phi_x + phi_m0, statements[:line_no], ignore=index_m0)
                phi_m1 = self.construct_phi(1, m1)

                c1 = int(question[4].split(' ')[0])
                c2 = int(question[4].split(' ')[1])
                if (index_m0 == c1 or index_m0 == c2) and (index_m1 == c1 or index_m1 == c2):
                    fake_correct_answers += 1

            if article_no <= 2:
                predicted, _ = self.find_word(phi_x + phi_m0 + phi_m1, verbose=False)
                print "%d, %d, %d: predicted: %s, correct: %s" % (i, article_no, line_no, self.id_to_word[predicted], self.id_to_word[correct])
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

    if len(sys.argv) > 2:
        n_epochs = int(sys.argv[2])
    else:
        n_epochs = 10

    memNN = MemNN(n_words=num_words, n_embedding=100, lr=0.01, n_epochs=n_epochs, margin=0.1, word_to_id=word_to_id)
    # memNN.train(train_dataset, train_questions, lr_schedule=dict([(0, 0.01), (20, 0.005), (50, 0.001)]))
    memNN.train(train_dataset, train_questions)
    memNN.predict(test_dataset, test_questions)
