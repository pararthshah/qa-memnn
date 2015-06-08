from nltk_utils import *

import nltk
from nltk.corpus import wordnet as wn

def memoizefirst(f):
    memo = {}
    def helper(x, y):
        if x not in memo:
            memo[x] = f(x, y)
        return memo[x]
    return helper

def get_noun_set(article, tokens):
    tags = nltk.pos_tag(tokens)
    nouns = set(
        map(
            lambda x: x[0],
            filter(
                lambda x: x[1] == wn.NOUN,
                map(lambda x: (x[0], penn_to_wn(x[1])), tags),
            )
        )
    )
    return nouns

get_noun_set = memoizefirst(get_noun_set)

def prune_statements(dataset, questions, debug=True):
    total_old = 0
    total_new = 0

    for i in range(len(questions)):
        question = questions[i]
        new_statements = []
        old_statements = question[2]

        # Keep only statements which have at least 1 common noun
        q = question[3]
        q_nouns = get_noun_set('|'.join(q), q)

        for s in old_statements:
            s_nouns = get_noun_set('|'.join(s), s)
            if len(s_nouns.intersection(q_nouns)) > 0:
                new_statements.append(s)

        questions[i][2] = new_statements
        total_old += len(old_statements)
        total_new += len(new_statements)

        if debug and i < 3:
            print "Question: ", q, "Statements:\n", old_statements, "\n", new_statements, "\nbefore %d after %d" % (len(old_statements), len(new_statements))

    #print("Before %d After %d" % (total_old, total_new))
    return questions
