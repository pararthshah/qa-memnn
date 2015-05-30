import re

from theano_util import *

def parse_qa_dataset(input_dir, word_id=0, word_to_id={}, update_word_ids=True):
    dataset = []
    questions = []
    with open(input_dir + '/question_answer_pairs.txt') as f:
        statements = []
        article_no = 0
        line_no = 0
        stmt_to_line = {}
        for line in f:
            # Skip first line
            if 'ArticleFile' in line:
                continue

            line = line.strip()

            # Skip empty lines
            if len(line) == 0:
                continue

            parts = line.split('\t')
            if len(parts) != 6:
                print("Malformed line: " + line)
                continue

            question = parts[1]
            answer = parts[2]
            article_name = parts[5]

            # There are other fields in the dataset, use them later if you want

            # This dataset has repeated questions. What to do?

            # Don't answer questions with more than 1 word answers
            if len(answer.split(' ')) > 1:
                # Skip for now
                continue

            question_parts = question.split('\t')
            tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0].strip()).split()
            if update_word_ids:
                for token in (tokens + [answer]):
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1

            # Get all statements in the dataset for this question

            ######## Need to change things below this.

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

