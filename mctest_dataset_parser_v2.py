import re
import sys, os
import cPickle

from theano_util import (
    pad_memories,
    pad_statement,
)

from pos_pruning import prune_statements

def only_words(line):
    ps = re.sub(r'[^a-zA-Z0-9\']', r' ', line)
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip().split(' ')
    return rs

def clean_sentence(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!\']', ' ', line) # Split on punctuations and hex characters
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs

def get_sentences(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!\']', ' ', line) # Split on punctuations and hex characters
    s = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\t', ps) # Split on sentences
    ws = re.sub(r'(\W)', r' \1 ', s) # Put spaces around punctuations
    ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
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

def transform_ques_weak(question, word_to_id, num_words):
    indices = []
    for stmt in question[2]:
        index_stmt = map(lambda x: word_to_id[x], stmt)
        indices.append(index_stmt)
    question[2] = indices
    question[3] = map(lambda x: word_to_id[x], question[3])
    question[5] = map(lambda l: map(lambda x: word_to_id[x], l), question[5])
    return question

def parse_mc_test_dataset(questions_file, answers_file, word_id=0, word_to_id={}, update_word_ids=True, pad=True, add_pruning=False):
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

            statements.append(tokens)
            dataset.append(tokens)

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

            # if len(answer) > 1:
            #     more_than_1_word_answers += 1
            #     continue

            if update_word_ids:
                for token in q_words:
                    if token not in word_to_id:
                        word_to_id[token] = word_id
                        word_id += 1
                for o in options:
                    for token in o:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1
            else:
                q_words = filter(lambda x: x in word_to_id, q_words)

            if q_words[0] == 'multiple' or q_words[0] == 'one':
                del q_words[0]

            # Ignore questions with unknown words in the answer
            options_word_ids = []
            skip = False
            for o in options:
                option_word_ids = []
                for w in o:
                    if w not in word_to_id:
                        if update_word_ids:
                            word_to_id[w] = word_id
                            word_id += 1
                            option_word_ids.append(w)
                        else:
                            skip = True
                            break
                    else:
                        option_word_ids.append(w)
                if skip:
                    break
                else:
                    #if len(option_word_ids) > 1:
                    #    skip = True
                    #    more_than_1_word_answers += 1
                    #    break
                    options_word_ids.append(option_word_ids)

            if skip:
                answer_word_unknown += 1
                continue

            article_no = len(questions)
            questions.append([article_no, -1, statements, q_words, correct, options_word_ids])

    print "There are %d questions" % len(questions)
    print "There are %d statements" % len(dataset)
    print "There are %d words" % len(word_to_id)
    print "Ignored %d questions which had more than 1 word answers" % more_than_1_word_answers
    print "Ignored %d questions which had an unknown answer word" % answer_word_unknown

    if add_pruning:
        print("Trying to prune extraneaous statements...")
        questions = prune_statements(dataset, questions)
        before_prune = len(questions)
        questions = filter(lambda x: len(x[2]) > 1, questions)
        after_prune = len(questions)
        print("Pruning invalidated %d questions" % (before_prune - after_prune))

    max_stmts = None
    max_words = None
    if pad:
        s_lens = []
        q_lens = []
        for i in xrange(len(questions)):
            q = questions[i]
            s_lens.append(len(q[2]))
            for j in xrange(len(q[2])):
                q_lens.append(len(q[2][j]))

        max_stmts = max(s_lens)
        max_words = max(q_lens)
        print "Max statement length: ", max_words
        print "Max number of statements: ", max_stmts

        for i in xrange(len(questions)):
            q = questions[i]
            # Statements

            for j in xrange(len(q[2])):
                q[2][j] = pad_statement(q[2][j], null_word, max_words)

            q[2] = pad_memories(q[2], null_word, max_stmts, max_words)
            q[3] = pad_statement(q[3], null_word, max_words)

            for j in xrange(len(q[5])):
                q[5][j] = pad_statement(q[5][j], null_word, max_words)


    print("Final processing...")
    questions_seq = map(lambda x: transform_ques_weak(x, word_to_id, word_id), questions)
    return dataset, questions_seq, word_to_id, word_id, null_word_id, max_stmts, max_words

def parse_stop_words(stop_file, word_id=0, word_to_id={}, update_word_ids=False):
    stop_words = set()
    with open(stop_file) as f:
        for line in f:
            token = line.strip()
            if not token in word_to_id:
                if update_word_ids:
                    word_to_id[token] = word_id
                    word_id += 1
                else:
                    continue
            stop_words.add(word_to_id[token])
    return stop_words

if __name__ == "__main__":
    ADD_PADDING = True
    ADD_PRUNING = False
    # Consider padding from the other side

    if len(sys.argv) > 2:
        dataset = sys.argv[2]
    else:
        dataset = 'mc160'

    train_file = dataset + '.train.tsv'
    print "Train file:", train_file

    train_answers = train_file.replace('tsv', 'ans')

    test_file = train_file.replace('train', 'test')
    test_answers = test_file.replace('tsv', 'ans')

    data_dir = sys.argv[1]

    train_obj = parse_mc_test_dataset(os.path.join(data_dir, train_file), os.path.join(data_dir, train_answers), pad=ADD_PADDING, add_pruning=ADD_PRUNING)
    num_words = train_obj[3]
    word_to_id = train_obj[2]
    test_obj = parse_mc_test_dataset(os.path.join(data_dir, test_file), os.path.join(data_dir, test_answers), word_id=num_words, word_to_id=word_to_id, update_word_ids=True, pad=ADD_PADDING, add_pruning=ADD_PRUNING)
    num_words = test_obj[3]
    word_to_id = test_obj[2]

    # Add dev to test
    # test2_file = train_file.replace('train', 'dev')
    # test2_answers = test2_file.replace('tsv', 'ans')
    # test2_obj = parse_mc_test_dataset(os.path.join(data_dir, test2_file), os.path.join(data_dir, test2_answers), word_id=num_words, word_to_id=word_to_id, update_word_ids=True, pad=ADD_PADDING, add_pruning=ADD_PRUNING)

    #test_obj[0] += test2_obj[0]
    #test_obj[1] += test2_obj[1]

    stop_file = 'stopwords.txt'
    stop_obj = parse_stop_words(os.path.join(data_dir, stop_file), word_id=num_words, word_to_id=word_to_id)

    # Pickle!!!!
    train_pickle = train_file.replace('tsv', 'pickle')
    print("Pickling train... " + train_pickle)
    f = file(os.path.join(data_dir, train_pickle), 'wb')
    cPickle.dump(train_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    test_pickle = test_file.replace('tsv', 'pickle')
    print("Pickling test... " + test_pickle)
    f = file(os.path.join(data_dir, test_pickle), 'wb')
    cPickle.dump(test_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    stop_pickle = stop_file.replace('txt', 'pickle')
    print("Pickling stop words... " + stop_pickle)
    f = file(os.path.join(data_dir, stop_pickle), 'wb')
    cPickle.dump(stop_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
