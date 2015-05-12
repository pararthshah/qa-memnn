import numpy as np
import theano
import theano.tensor as T
import sys, re, random

dtype=theano.config.floatX

def init_shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(dtype))

def init_shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=dtype))

class MemNN:
    def __init__(self, n_words=1000, n_embedding=100, lr=0.01, margin=0.1, n_epochs=100):
        self.n_embedding = n_embedding
        self.lr = lr
        self.margin = margin
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_D = 2 * self.n_words

        phi_x = T.vector('phi_x')
        phi_f1 = T.vector('phi_f1')
        phi_f1bar = T.vector('phi_f1bar')
        phi_f = T.vector('phi_f')
        
        self.U_O = init_shared_normal(n_embedding, self.n_D, 0.01)
        cost = self.calc_cost(phi_x, phi_f1, phi_f1bar)
        params = [self.U_O]
        gradient = T.grad(cost, params)

        updates=[]
        for param, gparam in zip(params, gradient):
            updates.append((param, param - gparam * self.lr))

        self.train_function = theano.function(inputs = [phi_x, phi_f1, phi_f1bar],
                                         outputs = cost,
                                         updates = updates)

        score = self.calc_score(phi_x, phi_f)
        self.predict_function = theano.function(inputs = [phi_x, phi_f], outputs = score)

    def calc_score(self, phi_x, phi_y):
        #return T.dot(T.dot(phi_x.T, self.U_O.T), T.dot(self.U_O, phi_y))
        return T.dot(self.U_O.dot(phi_x), self.U_O.dot(phi_y))

    def calc_cost(self, phi_x, phi_f1, phi_f1bar):
        correct_score = self.calc_score(phi_x, phi_f1)
        false_score = self.calc_score(phi_x, phi_f1bar)
        print correct_score.type
        cost = T.maximum(0, self.margin - correct_score + false_score)
        print theano.pp(cost)
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
            max_score = 0
            for i in range(line_no):
                phi_f = np.zeros((self.n_D,))
                phi_f[num_words:2*num_words] = dataset[article_no][i]
                
                #print phi_x, phi_f
                score = self.predict_function(phi_x, phi_f)
                if score > max_score:
                    max_score = score
                    answer = i

            if answer == correct_stmt:
                correct_answers += 1
            else:
                wrong_answers += 1

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)


def parse_dataset(input_file):
    dataset = []
    questions = []
    word_to_id = {}
    word_id = 0
    with open(input_file) as f:
        statements = []
        article_no = -1
        line_no = 0
        stmt_to_line = {}
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[:2] == '1 ' and len(statements) > 0: # new article
                dataset.append(statements)
                statements = []
                line_no = 0
                stmt_to_line = {}
                article_no += 1
            if '\t' in line:
                question_parts = line.split('\t')
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0]).split()
                for token in tokens[1:]:
                    if not token in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
                questions.append([article_no, line_no, ' '.join(tokens[1:]), question_parts[1], stmt_to_line[question_parts[2]]])
            else:
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                stmt_to_line[tokens[0]] = line_no
                for token in tokens[1:]:
                    if not token in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
                statements.append(' '.join(tokens[1:]))
                line_no += 1
    return dataset, questions, word_to_id, word_id

def compute_phi(input_str, word_to_id, num_words):
    phi = np.zeros((num_words,))
    for token in input_str.split():
        phi[word_to_id[token]] += 1
    return phi

def transform_ques(question, word_to_id, num_words):
    question[2] = compute_phi(question[2], word_to_id, num_words)
    return question

if __name__ == "__main__":
    training_dataset = sys.argv[1]
    test_dataset = training_dataset.replace('train', 'test')

    dataset, questions, word_to_id, num_words = parse_dataset(training_dataset)
    dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), dataset)
    questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), questions)
    # print dataset[0], dataset_bow[0], questions_bow[0]
    #print len(dataset_bow)
    memNN = MemNN(n_words=num_words, n_epochs=100, margin=1.0)
    memNN.train(dataset_bow, questions_bow, num_words)

    test_dataset, test_questions, _, _ = parse_dataset(test_dataset)
    test_dataset_bow = map(lambda y: map(lambda x: compute_phi(x, word_to_id, num_words), y), test_dataset)
    test_questions_bow = map(lambda x: transform_ques(x, word_to_id, num_words), test_questions)    
    memNN.predict(test_dataset_bow, test_questions_bow)
