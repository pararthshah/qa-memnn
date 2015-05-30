import numpy as np
import re, sys
import theano
import theano.tensor as T
from keras.utils.theano_utils import shared_zeros

dtype=theano.config.floatX

def init_shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(dtype))

def init_shared_normal_tensor(num_slices, num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_slices, num_rows, num_cols)).astype(dtype))

def init_shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=dtype))

def get_param_updates(params, grads, lr, method=None, **kwargs):
    rho = 0.95
    epsilon = 1e-6

    accumulators = [shared_zeros(p.get_value().shape) for p in params]
    updates=[]

    if method == 'adadelta':
        print "Using ADADELTA"
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            new_a = rho * a + (1 - rho) * g ** 2 # update accumulator
            updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + epsilon) / T.sqrt(new_a + epsilon)

            new_p = p - lr * update
            updates.append((p, new_p)) # apply constraints

            # update delta_accumulator
            new_d_a = rho * d_a + (1 - rho) * update ** 2
            updates.append((d_a, new_d_a))


    elif method == 'adam':
        # unimplemented
        print "Using ADAM"

    elif method == 'adagrad':
        print "Using ADAGRAD"
        for p, g, a in zip(params, grads, accumulators):
            new_a = a + g ** 2 # update accumulator
            updates.append((a, new_a))

            new_p = p - lr * g / T.sqrt(new_a + epsilon)
            updates.append((p, new_p)) # apply constraints

    else: # Default
        print "Using MOMENTUM"
        momentum = kwargs['momentum']
        for param, gparam in zip(params, grads):
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - param_update * lr))
            updates.append((param_update, momentum*param_update + (1. - momentum)*gparam))

    return updates


def compute_bow(input_str, word_to_id, num_words):
    bow = np.zeros((num_words,))
    for token in input_str.split():
        bow[word_to_id[token]] += 1
    return bow

def compute_seq(input_str, word_to_id, num_words):
    seq = []
    for token in input_str.split():
        seq.append(word_to_id[token])
    return seq

def transform_ques(question, word_to_id, num_words):
    question.append(compute_seq(question[2], word_to_id, num_words))
    question[2] = compute_bow(question[2], word_to_id, num_words)
    return question

def parse_dataset(input_file, word_id=0, word_to_id={}, update_word_ids=True):
    dataset = []
    questions = []
    with open(input_file) as f:
        statements = []
        article_no = 0
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
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0].strip()).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

                # To handle the case of "3 6"
                lines = None
                if ' ' in question_parts[2]:
                    stmts = question_parts[2].split(' ')
                    lines = ''
                    for stmt in stmts:
                        lines += str(stmt_to_line[stmt]) + ' '
                    lines = lines.strip()
                else:
                    lines = str(stmt_to_line[question_parts[2]])

                questions.append([article_no, line_no, ' '.join(tokens[1:]), word_to_id[question_parts[1]], lines])
            else:
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                stmt_to_line[tokens[0]] = line_no
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1
                statements.append(' '.join(tokens[1:]))
                line_no += 1
        if len(statements) > 0:
            dataset.append(statements)
    dataset_bow = map(lambda y: map(lambda x: compute_bow(x, word_to_id, word_id), y), dataset)
    dataset_seq = map(lambda y: map(lambda x: compute_seq(x, word_to_id, word_id), y), dataset)
    questions_bow = map(lambda x: transform_ques(x, word_to_id, word_id), questions)
    return dataset_seq, dataset_bow, questions_bow, word_to_id, word_id

def parse_dataset_weak(input_file, word_id=0, word_to_id={}, update_word_ids=True):
    dataset = []
    questions = []
    with open(input_file) as f:
        statements = []
        article_no = 0
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
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0].strip()).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

                questions.append([article_no, line_no, statements[:line_no] + [tokens[1:]], question_parts[1]])
            else:
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                stmt_to_line[tokens[0]] = line_no
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1
                statements.append(tokens[1:])
                line_no += 1
        if len(statements) > 0:
            dataset.append(statements)
    questions_seq = map(lambda x: transform_ques_weak(x, word_to_id, word_id), questions)
    return dataset, questions_seq, word_to_id, word_id

def transform_ques_weak(question, word_to_id, num_words):
    indices = []
    for stmt in question[2]:
        index_stmt = map(lambda x: word_to_id[x], stmt)
        indices.append(index_stmt)
    question[2] = indices
    question[3] = word_to_id[question[3]]
    return question

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    train_dataset, train_questions, word_to_id, num_words = parse_dataset_weak(train_file)
    test_dataset, test_questions, _, _ = parse_dataset_weak(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)

    # each element of train_questions contains: [article_no, line_no, [lists of indices of statements and question], index of answer word]
    print train_questions[0]
