import numpy as np
import sys, random, pprint
import cPickle
import math, os

class MCTestBaseline:
    def __init__(self, n_words=20, word_to_id=None, null_word_id=-1):
        self.n_words = n_words
        self.word_to_id = word_to_id
        self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())
        self.null_word_id = null_word_id

    def remove_nulls(self, stmt):
        return filter(lambda x: x != self.null_word_id, stmt)

    def compute_inverse_count(self, stmt_list):
        counts = {}
        for word in stmt_list:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1

        ic = {}
        for k, v in counts.iteritems():
            ic[k] = math.log10(1 + 1.0/float(v))
        return ic

    def compute_scores(self, statements, question, answers, stop_words):
        stmt_list = [word for stmt in statements for word in self.remove_nulls(stmt)]
        stmt_set = set(stmt_list)
        ques_set = set(self.remove_nulls(question))
        ans_set = map(lambda x: set(self.remove_nulls(x)), answers)
        ic = self.compute_inverse_count(stmt_list)
        scores = []
        for i in range(4):
            sw_score = -1
            S = ans_set[i] | ques_set
            S_list = list(S)
            for j in range(len(stmt_list)):
                curr_score = 0
                for w in range(len(S_list)):
                    if j+w < len(stmt_list) and stmt_list[j+w] in S:
                        if stmt_list[j+w] in stmt_set:
                            curr_score += ic[stmt_list[j+w]]
                if sw_score == -1 or curr_score > sw_score:
                    sw_score = curr_score

            d_score = -1
            S_Q = (ques_set & stmt_set) - stop_words
            S_A = (ans_set[i] & stmt_set) - stop_words
            if len(S_Q) == 0 or len(S_A) == 0:
                d_score = 1
            else:
                min_dist = len(stmt_list)
                last_q = -1
                last_a = -1
                for i in range(len(stmt_list)):
                    if stmt_list[i] in S_Q and stmt_list[i] in S_A:
                        min_dist = 0
                        break
                    if stmt_list[i] in S_Q:
                        last_q = i
                        if last_a >= 0 and i - last_a < min_dist:
                            min_dist = i - last_a
                    elif stmt_list[i] in S_A:
                        last_a = i
                        if last_q >= 0 and i - last_q < min_dist:
                            min_dist = i - last_q
                d_score = float(min_dist + 1) / float(len(stmt_list) + 1)
            scores.append(sw_score - d_score)

        return scores


    def train(self):
        pass

    def predict(self, dataset, questions, stop_words=set(), max_words=20, print_errors=True):
        correct_answers = 0
        wrong_answers = 0

        for i, question in enumerate(questions):
            statements_seq = question[2]
            question_seq = question[3]
            answers = question[5]
            correct = question[4]

            # print statements_seq
            # print question_seq
            # print answers
            # print correct
            

            scores = self.compute_scores(statements_seq, question_seq, answers, stop_words)
            predicted = np.argmax(scores)

            if predicted == correct:
                correct_answers += 1
            else:
                if print_errors and np.random.rand() < 0.1:
                    correct_words = map(lambda x: self.id_to_word[x], self.remove_nulls(question[5][correct]))
                    predicted_words = map(lambda x: self.id_to_word[x], self.remove_nulls(question[5][predicted]))
                    print 'Correct: %s (%d %.3f), Guess: %s (%d %.3f)' % (correct_words, correct, scores[correct], predicted_words, predicted, scores[predicted])
                wrong_answers += 1

            #if len(questions) > 1000:
            #    print '(%d/%d) %d correct, %d wrong' % (i+1, len(questions), correct_answers, wrong_answers)

        accuracy = 100.0 * float(correct_answers) / (correct_answers + wrong_answers)
        print '%d correct, %d wrong, %.2f%% acc' % (correct_answers, wrong_answers, accuracy)


if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')
    stop_file = os.path.join(os.path.dirname(train_file), 'stopwords.pickle')

    print("Loading pickled train dataset")
    f = file(train_file, 'rb')
    obj = cPickle.load(f)
    train_dataset, train_questions, word_to_id, num_words, null_word_id, train_max_stmts, train_max_words = obj
    f.close()

    print("Loading pickled test dataset")
    f = file(test_file, 'rb')
    obj = cPickle.load(f)
    test_dataset, test_questions, _, _, _, test_max_stmts, test_max_words = obj
    f.close()

    print("Loading pickled stop words")
    f = file(stop_file, 'rb')
    obj = cPickle.load(f)
    stop_words = obj
    f.close()

    print "Dataset has %d words" % num_words

    baseline = MCTestBaseline(n_words=num_words, word_to_id=word_to_id, null_word_id=null_word_id)
    baseline.predict(train_dataset, train_questions, stop_words, train_max_words)
    baseline.predict(test_dataset, test_questions, stop_words, test_max_words)
