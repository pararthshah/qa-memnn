import numpy as np
import re
import theano

dtype=theano.config.floatX

def init_shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(dtype))

def init_shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=dtype))

def parse_dataset(input_file):
    dataset = []
    questions = []
    word_to_id = {}
    word_id = 0
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
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0]).split()
                for token in tokens[1:]:
                    if not token in word_to_id:
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
                    lines = stmt_to_line[question_parts[2]]

                questions.append([article_no, line_no, ' '.join(tokens[1:]), question_parts[1], lines])
            else:
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                stmt_to_line[tokens[0]] = line_no
                for token in tokens[1:]:
                    if not token in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
                statements.append(' '.join(tokens[1:]))
                line_no += 1
        if len(statements) > 0:
            dataset.append(statements)
    return dataset, questions, word_to_id, word_id

def compute_phi(input_str, word_to_id, num_words):
    phi = np.zeros((num_words,))
    for token in input_str.split():
        phi[word_to_id[token]] += 1
    return phi

def transform_ques(question, word_to_id, num_words):
    question[2] = compute_phi(question[2], word_to_id, num_words)
    return question

