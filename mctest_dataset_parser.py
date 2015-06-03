import re

from theano_util import *

def only_words(line):
    ps = re.sub(r'[^a-zA-Z0-9]', r' ', line)
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ns) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs

def clean_sentence(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!]', ' ', line) # Split on punctuations and hex characters
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ns) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs

def get_sentences(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!]', ' ', line) # Split on punctuations and hex characters
    s = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\t', ps) # Split on sentences
    ws = re.sub(r'(\W)', r' \1 ', s) # Put spaces around punctuations
    ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ns) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs.split('\t')

def get_answer_index(a):
    answer_to_index = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    return answer_to_index[a]

def parse_mc_test_dataset(questions_file, answers_file, word_id=0, word_to_id={}, update_word_ids=True, max_stmts=20, max_words=20, pad=True):
    dataset = []
    questions = []

    null_word = '<NULL>'
    if null_word not in word_to_id:
        if update_word_ids == True:
            word_to_id[null_word] = word_id
            word_id += 1
        else:
            print "Null word not found!! AAAAA"
            sys.exit(1)
    null_word_id = word_to_id[null_word]

    article_files = set()
    print("Parsing questions %s %s" % (questions_file, answers_file))
    q_file = open(questions_file, 'r')
    a_file = open(answers_file, 'r')

    questions_data = q_file.readlines()
    answers_data = a_file.readlines()

    assert(len(questions_data) == len(answers_data))

    more_than_1_word_answers = 0
    answer_word_unknown = 0

    for i in xrange(len(questions_data)):
        question_line = questions_data[i]
        answer_line = answers_data[i]

        question_pieces = question_line.strip().split('\t')
        assert(len(question_pieces) == 23)

        answer_pieces = answer_line.strip().split('\t')
        assert(len(answer_pieces) == 4)

        text = question_pieces[2]
        text = text.replace('\\newline', ' ')
        sentences = get_sentences(text)

        statements = []
        for s in sentences:
            tokens = s.strip().split()

            if update_word_ids:
                for token in tokens:
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
            else:
                tokens = filter(lambda x: x in word_to_id, tokens)

            if pad:
                tokens = pad_statement(tokens, null_word, max_words)

            statements.append(tokens)
            dataset.append(tokens)

        if pad:
            statements = pad_memories(statements, null_word, max_stmts, max_words)

        # 4 questions
        for j in range(4):
            q_index = (j * 5) + 3
            q_words = question_pieces[q_index]
            q_words = clean_sentence(q_words).split()

            options = [
                only_words(question_pieces[q_index + 1]),
                only_words(question_pieces[q_index + 2]),
                only_words(question_pieces[q_index + 3]),
                only_words(question_pieces[q_index + 4]),
            ]
            correct = get_answer_index(answer_pieces[j])
            answer = options[correct]

            if update_word_ids:
                for token in (q_words + options):
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
            else:
                q_words = filter(lambda x: x in word_to_id, q_words)

            if pad:
                q_words = pad_statement(q_words, null_word, max_words)

            # Ignore more than 1 word answers
            if len(answer.split(' ')) > 1:
                more_than_1_word_answers += 1
                continue
            elif len(filter(lambda x: x not in word_to_id, options)) > 0:
                answer_word_unknown += 1
                continue

            option_word_ids = map(lambda x: word_to_id[x], options)

            article_no = len(questions)
            questions.append([article_no, -1, statements, q_words, answer, option_word_ids])

    print "There are %d questions" % len(questions)
    print "There are %d statements" % len(dataset)
    print "There are %d words" % len(word_to_id)
    print "Ignored %d questions which had more than 1 word answers" % more_than_1_word_answers
    print "Ignored %d questions which had an unknown answer word" % answer_word_unknown

    print("Final processing...")
    questions_seq = map(lambda x: transform_ques_weak(x, word_to_id, word_id), questions)
    return dataset, questions_seq, word_to_id, word_id, null_word_id

import cPickle

if __name__ == "__main__":
    ADD_PADDING = True

    train_file = 'mc500.train.tsv'
    train_answers = train_file.replace('tsv', 'ans')

    test_file = train_file.replace('train', 'test')
    test_answers = test_file.replace('tsv', 'ans')

    data_dir = sys.argv[1]

    train_dataset, train_questions, word_to_id, num_words, null_word_id = parse_mc_test_dataset(data_dir + '/' + train_file, data_dir + '/' + train_answers, pad=ADD_PADDING)
    test_dataset, test_questions, word_to_id, num_words, null_word_id = parse_mc_test_dataset(data_dir + '/' + test_file, data_dir + '/' + test_answers, word_id=num_words, word_to_id=word_to_id, update_word_ids=False, pad=ADD_PADDING)

    # Add dev to test
    test2_file = train_file.replace('train', 'dev')
    test2_answers = test2_file.replace('tsv', 'ans')
    test2_dataset, test2_questions, word_to_id, num_words, null_word_id = parse_mc_test_dataset(data_dir + '/' + test2_file, data_dir + '/' + test2_answers, word_id=num_words, word_to_id=word_to_id, update_word_ids=False, pad=ADD_PADDING)

    test_dataset += test2_dataset
    test_questions += test2_questions

    # Pickle!!!!
    print("Pickling train...")
    train_pickle = train_file.replace('tsv', 'pickle')
    f = file(data_dir + '/' + train_pickle, 'wb')
    cPickle.dump((train_dataset, train_questions, word_to_id, num_words, null_word_id), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    print("Pickling test...")
    test_pickle = test_file.replace('tsv', 'pickle')
    f = file(data_dir + '/' + test_pickle, 'wb')
    cPickle.dump((test_dataset, test_questions, word_to_id, num_words, null_word_id), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
