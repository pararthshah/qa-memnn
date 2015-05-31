import re

from theano_util import *

def parse_qa_dataset(input_dir, word_id=0, word_to_id={}, update_word_ids=True):
    dataset = []
    questions = []
    with open(input_dir + '/question_answer_pairs.txt') as f:
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
            question_tokens = tokens
            if update_word_ids:
                for token in (tokens + [answer]):
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1

            # Get all statements in the dataset for this question

            article_file = input_dir + '/' + article_name + '.txt'

            s_file = open(article_file)
            statements = []
            for statement in s_file:
                if len(statement.strip()) == 0:
                    continue

                tokens = re.sub(r'([\.\?])$', r' \1', statement.strip()).split()

                if len(tokens) == 0:
                    continue

                article = tokens
                statements.append(article)
                if update_word_ids:
                    for token in tokens:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

            article_no = len(dataset)
            if len(statements) == 0:
                continue

            dataset.append(statements)
            questions.append([article_no, -1, statements + [question_tokens], answer])

    questions_seq = map(lambda x: transform_ques_weak(x, word_to_id, word_id), questions)
    return dataset, questions_seq, word_to_id, word_id

if __name__ == "__main__":
    train_file = sys.argv[1]

    train_dataset, train_questions, word_to_id, num_words = parse_qa_dataset(train_file)
    #test_dataset, test_questions, _, _ = parse_dataset_weak(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False)

    # each element of train_questions contains: [article_no, line_no, [lists of indices of statements and question], index of answer word]
    print train_questions[0]
