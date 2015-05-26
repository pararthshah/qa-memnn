from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys, re

from keras.preprocessing import sequence
from keras.initializations import uniform
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

# mode can be 'baseline' or 'memnn'
def load_dataset(input_file, word_id=0, word_to_id={}, update_word_ids=True, mode='memnn'):
    #dataset = []
    dataset_ids = []
    #labels = []
    label_ids = []
    with open(input_file) as f:
        article = {}
        article_no = 0
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[:2] == '1 ' and len(dataset_ids) > 0: # new article
                article = {}
                article_no += 1
            if '\t' in line: # question
                question_parts = line.split('\t')
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0].strip()).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1
                    if question_parts[1] not in word_to_id:
                            word_to_id[question_parts[1]] = word_id
                            word_id += 1

                stmt_ids = map(int, question_parts[2].strip().split())
                sequence = []
                if mode == 'baseline':
                    for s in range(int(tokens[0])):
                        if s in article:
                            sequence += article[s]
                else:
                    for s in stmt_ids:
                        sequence += article[s]

                for token in tokens[1:]:
                    sequence.append(token)

                if article_no == 0:
                    print("seq: %s | label: %s" % (' '.join(sequence).ljust(70), question_parts[1]))

                dataset_ids.append(map(lambda t: word_to_id[t], sequence))
                label_ids.append(word_to_id[question_parts[1]])

            else: # statement
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

                line_no = int(tokens[0])
                article[line_no] = []
                for token in tokens[1:]:
                    article[line_no].append(token)

    return dataset_ids, label_ids, word_to_id, word_id

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    mode = 'memnn'
    if len(sys.argv) > 2:
        mode = sys.argv[2] # should be 'baseline' or 'memnn'

    nb_epoch = 10
    if len(sys.argv) > 3:
        nb_epoch = int(sys.argv[3])

    print("Loading train data...")
    X_train, y_train, word_to_id, num_words = load_dataset(train_file, mode=mode)
    print("Loading test data...")
    X_test, y_test, _, _ = load_dataset(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False, mode=mode)

    id_to_word = dict([(v, k) for k, v in word_to_id.iteritems()])

    y_train_cat = np_utils.to_categorical(y_train, nb_classes=num_words)
    y_test_cat = np_utils.to_categorical(y_test, nb_classes=num_words)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train)
    X_test = sequence.pad_sequences(X_test)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    batch_size = 1
    in_embedding_size = 100
    out_embedding_size = 100

    model = Sequential()
    model.add(Embedding(num_words, in_embedding_size))
    model.add(LSTM(in_embedding_size, out_embedding_size))
    model.add(Dropout(0.5))
    model.add(Dense(out_embedding_size, num_words))
    model.add(Activation('softmax'))

    sgd_optimizer = SGD(lr=0.006, momentum=0.9, decay=0.99, nesterov=True)
    adg_optimizer = Adagrad()
    rms_optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms_optimizer, class_mode="categorical", theano_mode='FAST_COMPILE')

    print("Train...")
    model.fit(X_train, y_train_cat, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, show_accuracy=True)
    score = model.evaluate(X_test, y_test_cat, batch_size=batch_size)
    print('Test score:', score)

    classes_proba = model.predict_proba(X_test, batch_size=batch_size)
    for i in range(5):
        probs = sorted(zip(range(len(classes_proba)), classes_proba[i].tolist()), key=lambda x: x[1], reverse=True)
        print('Test sample %d (Correct label: %s)' % (i, id_to_word[y_test[i]]))
        for j, p in probs[:5]:
            print(id_to_word[j].ljust(20) + ': ' + str(p))

    classes = np_utils.probas_to_classes(classes_proba)
    acc = np_utils.accuracy(classes, y_test)
    print('Test accuracy:', acc)

    # print(classes.shape)
    # print(classes[0])
    # print(y_test[0])

    # classes_list = classes.tolist()
    # print(map(lambda x: id_to_word[x], classes_list[:25]))
