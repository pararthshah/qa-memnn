from gensim.models import Word2Vec
import numpy

def prune_statements(dataset, questions):
    total_old = 0
    total_new = 0

    wvs = Word2Vec(dataset, min_count=0)

    for i in range(len(questions)):
        question = questions[i]
        new_statements = []
        old_statements = question[2][:-1]

        # Use word vectors and keep only the top 5

        sims = []
        q = question[2][-1]
        for s in old_statements:
            sims.append(wvs.n_similarity(q,s))

        sims2 = map(lambda x: x if type(x) is numpy.float64 else 0.0, sims)
        top = sorted(range(len(sims2)), key=sims2.__getitem__, reverse=True)
        new_statements = map(lambda x: old_statements[x], top[:5])

        questions[i][2] = new_statements
        total_old += len(old_statements)
        total_new += len(new_statements)
        #print("Question: ", questions[i][2][-1], " before %d after %d" % (len(old_statements), len(new_statements)))

    print("Before %d After %d" % (total_old, total_new))
    return questions
