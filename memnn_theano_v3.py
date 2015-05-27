import numpy as np
import theano
import theano.tensor as T
import sys, random, pprint

from theano_util import *
from keras.activations import tanh, hard_sigmoid
from keras.initializations import glorot_uniform, orthogonal
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix

def inspect_inputs(i, node, fn):
    print i, node, "inputs:", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print i, node, "outputs:", [output[0] for output in fn.outputs]

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100, momentum=0.9, word_to_id=None):
        self.n_embedding = n_embedding
        self.n_lstm_embed = n_embedding
        self.lr = lr
        self.momentum = momentum
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 3 * self.n_words + 3

        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

        # Question
        x = T.vector('x')
        phi_x = T.vector('phi_x')

        # True statements
        phi_f1_1 = T.vector('phi_f1_1')
        phi_f2_1 = T.vector('phi_f2_1')

        # False statements
        phi_f1_2 = T.vector('phi_f1_2')
        phi_f2_2 = T.vector('phi_f2_2')

        # Supporting memories
        m0 = T.vector('m0')
        m1 = T.vector('m1')
        phi_m0 = T.vector('phi_m0')
        phi_m1 = T.vector('phi_m1')

        # True word
        r = T.vector('r')

        # Scoring function
        self.U_O = init_shared_normal(n_embedding, self.n_D, 0.01)

        # LSTM
        self.W_i = glorot_uniform((self.n_words, self.n_words))
        self.U_i = orthogonal((self.n_words, self.n_words))
        self.b_i = shared_zeros((self.n_words))

        self.W_f = glorot_uniform((self.n_words, self.n_words))
        self.U_f = orthogonal((self.n_words, self.n_words))
        self.b_f = shared_zeros((self.n_words))

        self.W_c = glorot_uniform((self.n_words, self.n_words))
        self.U_c = orthogonal((self.n_words, self.n_words))
        self.b_c = shared_zeros((self.n_words))

        self.W_o = glorot_uniform((self.n_words, self.n_words))
        self.U_o = orthogonal((self.n_words, self.n_words))
        self.b_o = shared_zeros((self.n_words))

        mem_cost = self.calc_cost(phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, phi_m0)

        lstm_output = self.lstm_cost(x, m0, m1)
        self.predict_function_r = theano.function(inputs = [x, m0, m1], outputs = lstm_output)

        lstm_cost = -T.sum(T.mul(r, T.log(lstm_output)))

        cost = mem_cost + lstm_cost

        params = [
            self.U_O,
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o,
        ]

        gradient = T.grad(cost, params)

        l_rate = T.scalar('l_rate')

        # Parameter updates
        updates=[]
        for param, gparam in zip(params, gradient):
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - param_update * l_rate))
            updates.append((param_update, self.momentum*param_update + (1. - self.momentum)*gparam))

        # Theano functions
        self.train_function = theano.function(
            inputs = [x, phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2,
                      m0, m1, phi_m0, phi_m1, r,
                      theano.Param(l_rate, default=self.lr)],
            outputs = cost,
            updates = updates,
            on_unused_input='warn',
            )
            #mode='FAST_COMPILE')
            #mode='DebugMode')
            #mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs))

        # Candidate statement for prediction
        phi_f = T.vector('phi_f')

        score_o = self.calc_score_o(phi_x, phi_f)
        self.predict_function_o = theano.function(inputs = [phi_x, phi_f], outputs = score_o)

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = hard_sigmoid(xi_t + T.dot(h_tm1, u_i))
        f_t = hard_sigmoid(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * tanh(xc_t + T.dot(h_tm1, u_c))
        o_t = hard_sigmoid(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * tanh(c_t)
        return h_t, c_t

    def _lstm_fn(self, X):
        xi = T.dot(self.W_i, X.T) + self.b_i
        xf = T.dot(self.W_f, X.T) + self.b_f
        xc = T.dot(self.W_c, X.T) + self.b_c
        xo = T.dot(self.W_o, X.T) + self.b_o
        return xi, xf, xc, xo

    def lstm_cost(self, x, m0, m1):
        s1 = self._lstm_fn(x)
        s2 = self._lstm_fn(m0)
        s3 = self._lstm_fn(m1)
        seq0 = T.stack(s1[0], s2[0], s3[0])
        seq1 = T.stack(s1[1], s2[1], s3[1])
        seq2 = T.stack(s1[2], s2[2], s3[2])
        seq3 = T.stack(s1[3], s2[3], s3[3])

        # Since we have only 3 inputs always, maybe we shouldn't use scan
        # and do it directly.
        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[
                dict(input = seq0),
                dict(input = seq1),
                dict(input = seq2),
                dict(input = seq3),
            ],
            outputs_info=[
                alloc_zeros_matrix(self.n_words),
                alloc_zeros_matrix(self.n_words)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=-1
        )

        return T.nnet.softmax(outputs[-1])

    def calc_score_o(self, phi_x, phi_y_yp_t):
        return T.dot(self.U_O.dot(phi_x), self.U_O.dot(phi_y_yp_t))

    # phi_f1_1 = phi_f1 - phi_f1bar + phi_t1_1
    # phi_f1_2 = phi_f1bar - phi_f1 + phi_t1_2
    def calc_cost(self, phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2, phi_m0):
        score1_1 = self.calc_score_o(phi_x, phi_f1_1)
        score1_2 = self.calc_score_o(phi_x, phi_f1_2)

        score2_1 = self.calc_score_o(phi_x + phi_m0, phi_f2_1)
        score2_2 = self.calc_score_o(phi_x + phi_m0, phi_f2_2)

        s_o_cost = (
            T.maximum(0, self.margin - score1_1) + T.maximum(0, self.margin + score1_2) +
            T.maximum(0, self.margin - score2_1) + T.maximum(0, self.margin + score2_2)
        )

        return s_o_cost

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

    def make_one_hot(self, index):
        v = np.zeros((self.n_words))
        v[index] = 1.0
        return v

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
                x = question_phi
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
                r = self.make_one_hot(correct_word)

                cost = self.train_function(x, phi_x, phi_f1_1, phi_f1_2, phi_f2_1, phi_f2_2,
                                           m0, m1, phi_m0, phi_m1, r,
                                           l_rate)
                #print "%d: %f" % (i, cost)
                costs.append(cost)

            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def find_word(self, x, m0, m1):
        probs = self.predict_function_r(x, m0, m1)
        return np.argmax(probs)

    def predict(self, dataset, questions):
        correct_answers = 0
        wrong_answers = 0
        fake_correct_answers = 0
        for i, question in enumerate(questions):
            article_no = question[0]
            line_no = question[1]
            question_phi = question[2]
            correct = question[3]

            x = question_phi
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

                c1 = int(question[4].split(' ')[0])
                c2 = int(question[4].split(' ')[1])
                if (index_m0 == c1 or index_m0 == c2) and (index_m1 == c1 or index_m1 == c2):
                    fake_correct_answers += 1

            predicted = self.find_word(x, m0, m1)
            print 'Correct: %s (%d), Guess: %s (%d)' % (self.id_to_word[correct], correct, self.id_to_word[predicted], predicted)
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
    #memNN.predict(train_dataset, train_questions)
    memNN.predict(test_dataset, test_questions)
