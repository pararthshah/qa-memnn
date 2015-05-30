import numpy as np
import theano
import theano.tensor as T
import sys, random, pprint

from theano_util import *
from keras.activations import tanh, hard_sigmoid
from keras.initializations import glorot_uniform, orthogonal
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix

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

        # Question
        q = T.ivector('q')

        # True word
        r = T.iscalar('r')

        # Question embedding
        self.B = init_shared_normal(self.n_words, self.n_embedding, 0.01)

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
        self.weights = init_shared_normal_tensor(4, self.n_words, self.n_embedding, 0.01)

        # Final outut weight matrix
        self.W = init_shared_normal(self.n_embedding, self.n_words, 0.01)

        memory_cost = self.memnn_cost(x, q)
        memory_loss = -T.log(memory_cost[r]) # cross entropy on softmax

        cost = memory_loss

        params = [
            self.B,
            self.weights,
            #self.A1, self.C1,
            #self.C2,
            #self.C3,
            self.W,
        ]

        grads = T.grad(cost, params)

        # Parameter updates
        updates = get_param_updates(params, grads, lr=self.lr, method='adagrad')

        self.train_function = theano.function(
            inputs = [
                x, q, r
            ],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True,
            #mode='FAST_COMPILE'
            #mode='DebugMode'
        )

        self.predict_function = theano.function(
            inputs = [
                x, q
            ],
            outputs = cost,
            allow_input_downcast=True
        )

    def add_statement(self, statement, memories):
        # statement: question as a series of indices
        # memories: List of 6 lists, each is a list of memories
        # Append to memories?

        for i in range(len(self.weights)):
            z = T.sum(self.weights[i][statement], axis=0)
            memories[i] = T.stacklists([memories[i], [z]])

        return memories

    def _compute_memories(self, statement, previous, weights):
        memories = T.sum(weights[statement], axis=0)
        return memories

    def memnn_cost(self, statements, question):
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
            ],
            truncate_gradient = -1,
        )

        memories = T.stacklists(computed_memories).dimshuffle(1, 0, 2)

        # Embed question
        u1 = T.sum(self.B[question], axis=0)

        # Layer 1
        p = T.nnet.softmax(T.dot(memories[0], u1))
        p = T.repeat(T.reshape(p, (p.shape[0], 1)), memories[1].shape[0])
        o1 = T.sum(T.dot(p, memories[1]), axis=0)

        # Layer 2
        u2 = o1 + u1
        p = T.nnet.softmax(T.dot(memories[2], u2))
        p = T.repeat(T.reshape(p, (p.shape[0], 1)), memories[3].shape[0])
        o2 = T.sum(T.dot(p, memories[3]), axis=0)

        # Layer 3
        u3 = o2 + u2
        p = T.nnet.softmax(T.dot(memories[4], u3))
        p = T.repeat(T.reshape(p, (p.shape[0], 1)), memories[5].shape[0])
        o3 = T.sum(T.dot(p, memories[5]), axis=0)

        # Final
        output = T.nnet.softmax(T.dot(self.W, o3 + u3))

        return output[0]

    def train(self, dataset, questions, n_epochs=100, lr_schedule=None):
        l_rate = self.lr
        for epoch in xrange(n_epochs):
            costs = []

            if lr_schedule != None and epoch in lr_schedule:
                l_rate = lr_schedule[epoch]

            random.shuffle(questions)
            for i, question in enumerate(questions):
                article_no = question[0]
                article = dataset[article_no]
                line_no = question[1]
                statements_seq = question[2][:-1]
                question_seq = question[2][-1]

                if line_no <= 1:
                    continue

                # Correct word
                correct_word = question[3]

                cost = self.train_function(statements_seq, question_seq, correct_word)

                #print "%d: %f" % (i, cost)
                costs.append(cost)

            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def predict(self, dataset, questions):
        correct_answers = 0
        wrong_answers = 0
        for i, question in enumerate(questions):
            article_no = question[0]
            article = dataset[article_no]
            line_no = question[1]
            statements_seq = question[2][:-1]
            question_seq = question[2][-1]
            correct = question[3]

            predicted = self.predict_function(
                np.asarray(statements_seq), np.asarray(question_seq)
            )
            # print 'Correct: %s (%d), Guess: %s (%d)' % (self.id_to_word[correct], correct, self.id_to_word[predicted], predicted)
            if predicted == correct:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    train_dataset, train_questions, word_to_id, num_words = parse_dataset_weak(train_file)
    test_dataset, test_questions, _, _ = parse_dataset_weak(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)

    if len(sys.argv) > 2:
        n_epochs = int(sys.argv[2])
    else:
        n_epochs = 10

    wmemNN = WMemNN(n_words=num_words, n_embedding=100, lr=0.01, word_to_id=word_to_id)
    #memNN.train(train_dataset_seq, train_dataset_bow, train_questions, n_epochs=n_epochs, lr_schedule=dict([(0, 0.02), (20, 0.01), (50, 0.005), (80, 0.002)]))
    #memNN.train(train_dataset_seq, train_dataset_bow, train_questions, lr_schedule=dict([(0, 0.01), (15, 0.009), (30, 0.007), (50, 0.005), (60, 0.003), (85, 0.001)]))
    #memNN.train(train_dataset_seq, train_dataset_bow, train_questions)
    #memNN.predict(train_dataset, train_questions)
    #memNN.predict(test_dataset_seq, test_dataset_bow, test_questions)

    for i in xrange(20):
        wmemNN.train(train_dataset, train_questions, n_epochs=5)
        wmemNN.predict(test_dataset, test_questions)
