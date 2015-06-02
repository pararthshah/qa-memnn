import numpy as np
import theano
import theano.tensor as T
import sys, random, pprint

from theano_util import *
from keras.activations import tanh, hard_sigmoid
from keras.initializations import glorot_uniform, orthogonal
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
from keras.preprocessing import sequence

from qa_dataset_parser import parse_qa_dataset

theano.config.exception_verbosity = 'high'
# theano.config.allow_gc = False
#theano.config.profile = True

def inspect_inputs(i, node, fn):
    print i, node, "inputs:", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print i, node, "outputs:", [output[0] for output in fn.outputs]

class WMemNN:
    def __init__(self, n_words, n_embedding=100, lr=0.01, momentum=0.9, word_to_id=None):
        self.n_embedding = n_embedding
        self.lr = lr
        self.momentum = momentum
        self.n_words = n_words

        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())

        # Statement
        x = T.imatrix('x')

        # Positional encoding matrix
        pe = T.tensor3('pe')

        # Question
        q = T.ivector('q')

        # True word
        r = T.iscalar('r')

        # Question embedding
        self.B = glorot_uniform((self.n_words, self.n_embedding))

        # Statement input, output embeddings
        # self.A1 = init_shared_normal(self.n_words, self.n_embedding, 0.01)
        # self.C1 = init_shared_normal(self.n_words, self.n_embedding, 0.01)
        # self.A2 = self.C1
        # self.C2 = init_shared_normal(self.n_words, self.n_embedding, 0.01)
        # self.A3 = self.C2
        # self.C3 = init_shared_normal(self.n_words, self.n_embedding, 0.01)
        # self.weights = [
        #     self.A1,
        #     self.C1,
        #     #self.A2,
        #     self.C2,
        #     #self.A3,
        #     self.C3
        # ]
        self.weights = glorot_uniform((4, self.n_words, self.n_embedding))

        # Final outut weight matrix
        self.W = glorot_uniform((self.n_embedding, self.n_words))

        # Linear mapping between layers
        self.H = glorot_uniform((self.n_embedding, self.n_embedding))

        memory_cost = self.memnn_cost(x, q, pe)
        memory_loss = -T.log(memory_cost[r]) # cross entropy on softmax

        cost = memory_loss

        params = [
            self.B,
            self.weights,
            #self.A1, self.C1,
            #self.C2,
            #self.C3,
            self.W,
            self.H,
        ]

        grads = T.grad(cost, params)

        # Parameter updates
        updates = get_param_updates(params, grads, lr=self.lr, method='momentum', momentum=0.9)

        self.train_function = theano.function(
            inputs = [
                x, q, r, pe
            ],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True,
            #mode='FAST_COMPILE'
            #mode='DebugMode'
            #mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs)
            on_unused_input='warn'
        )

        self.predict_function = theano.function(
            inputs = [
                x, q, pe
            ],
            outputs = memory_cost,
            allow_input_downcast=True,
            on_unused_input='warn'
        )

    def _compute_memories(self, statement, previous, weights, pe_matrix):
        pe_weights = pe_matrix * weights[statement]
        memories = T.sum(pe_weights, axis=0)
        # memories = T.sum(weights[statement], axis=0)
        return memories

    def get_PE_matrix(self, num_words, embedding_size):
        pe_matrix = np.zeros((num_words, 4, embedding_size))
        for j in range(num_words):
            for k in range(embedding_size):
                value = (1 - float(j+1)/num_words) - (float(k+1)/embedding_size) * (1 - 2*float(j+1)/num_words)
                for i in range(4):
                    pe_matrix[j,i,k] = value
        return pe_matrix

    def memnn_cost(self, statements, question, pe_matrix):
        # statements: list of list of word indices
        # question: list of word indices

        computed_memories, updates = theano.scan(
            self._compute_memories,
            sequences = [statements],
            outputs_info = [
                alloc_zeros_matrix(self.weights.shape[0], self.n_embedding)
            ],
            non_sequences = [
                self.weights.dimshuffle(1, 0, 2),
                pe_matrix
            ],
            truncate_gradient = -1,
        )

        memories = T.stacklists(computed_memories).dimshuffle(1, 0, 2)

        # Embed question
        u1 = T.sum(self.B[question], axis=0)

        # Layer 1
        p = T.nnet.softmax(T.dot(u1, memories[0].T))
        o1 = T.dot(p, memories[1])

        # Layer 2
        u2 = o1 + T.dot(u1, self.H)
        p = T.nnet.softmax(T.dot(u2, memories[1].T))
        o2 = T.dot(p, memories[2])

        # Layer 3
        u3 = o2 + T.dot(u2, self.H)
        p = T.nnet.softmax(T.dot(u3, memories[2].T))
        o3 = T.dot(p, memories[3])

        # Final
        output = T.nnet.softmax(T.dot(o3 + u3, self.W))

        return output[0]

    def train(self, dataset, questions, n_epochs=100, lr_schedule=None, start_epoch=0):
        l_rate = self.lr
        for epoch in xrange(start_epoch, start_epoch + n_epochs):
            costs = []

            if lr_schedule != None and epoch in lr_schedule:
                l_rate = lr_schedule[epoch]

            random.shuffle(questions)
            for i, question in enumerate(questions):
                statements_seq = sequence.pad_sequences(np.asarray(question[2][:-1]))
                question_seq = np.asarray(question[2][-1])

                pe_matrix = self.get_PE_matrix(statements_seq.shape[1], self.n_embedding)

                # Correct word
                correct_word = question[3]

                cost = self.train_function(
                    statements_seq,
                    question_seq,
                    correct_word,
                    pe_matrix
                )

                #print "Epoch %d, sample %d: %f" % (epoch, i, cost)
                costs.append(cost)

            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def predict(self, dataset, questions):
        correct_answers = 0
        wrong_answers = 0
        for i, question in enumerate(questions):
            statements_seq = sequence.pad_sequences(np.asarray(question[2][:-1]))
            question_seq = np.asarray(question[2][-1])
            correct = question[3]

            pe_matrix = self.get_PE_matrix(statements_seq.shape[1], self.n_embedding)

            probs = self.predict_function(
                statements_seq, question_seq, pe_matrix
            )
            predicted = np.argmax(probs)

            if predicted == correct:
                correct_answers += 1
            else:
                #print 'Correct: %s (%d %.3f), Guess: %s (%d %.3f)' % (self.id_to_word[correct], correct, probs[correct], self.id_to_word[predicted], predicted, probs[predicted])
                wrong_answers += 1

            if len(questions) > 1000:
                print '(%d/%d) %d correct, %d wrong' % (i+1, len(questions), correct_answers, wrong_answers)

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    if len(sys.argv) > 2:
        n_epochs = int(sys.argv[2])
    else:
        n_epochs = 10

    mode = 'babi' # babi or wiki

    if '.pickle' in train_file:
        mode = 'wiki'

    if mode == 'babi':
        train_dataset, train_questions, word_to_id, num_words = parse_dataset_weak(train_file)
        test_dataset, test_questions, _, _ = parse_dataset_weak(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)
    elif mode == 'wiki':
        # Check for pickled dataset
        print("Loading pickled train dataset")
        f = file(train_file, 'rb')
        import cPickle
        obj = cPickle.load(f)
        train_dataset, train_questions, word_to_id, num_words = obj

        print("Loading pickled test dataset")
        f = file(test_file, 'rb')
        obj = cPickle.load(f)
        test_dataset, test_questions, word_to_id, num_words = obj
    elif mode == 'debug':
        train_dataset = []
        train_questions = [[0, 2, [[0, 1, 2, 3, 4, 5], [6, 7, 2, 3, 8, 5], [9, 10, 0, 11]], 4]]
        num_words = 12
        word_to_id = {}

    # print "Dataset has %d words" % num_words
    wmemNN = WMemNN(n_words=num_words, n_embedding=100, lr=0.01, word_to_id=word_to_id)

    for i in xrange(n_epochs/5):
        wmemNN.train(train_dataset, train_questions, n_epochs=5, start_epoch=5*i)
        wmemNN.predict(train_dataset, train_questions)
        wmemNN.predict(test_dataset, test_questions)
