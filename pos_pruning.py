from nltk_utils import *

import nltk
from nltk.corpus import wordnet as wn

def get_noun_set(tokens):
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

def prune_statements(dataset, questions):
    total_old = 0
    total_new = 0

    wvs = Word2Vec(dataset, min_count=0)

    for i in range(len(questions)):
        question = questions[i]
        new_statements = []
        old_statements = question[2][:-1]

        # Keep only statements which have at least 1 common noun
        q = question[2][-1]
        q_nouns = get_noun_set(q)

        for s in old_statements:
            s_nouns = get_noun_set(s)
            if len(s_nouns.intersection(q_nouns) > 0):
                new_statements.append(s)

        questions[i][2] = new_statements
        total_old += len(old_statements)
        total_new += len(new_statements)
        #print("Question: ", questions[i][2][-1], " before %d after %d" % (len(old_statements), len(new_statements)))

    print("Before %d After %d" % (total_old, total_new))
    return questions
