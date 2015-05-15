import numpy as np
import re

def parse_dataset(input_file, word_id=0, word_to_id={}, update_word_ids=True):
    dataset = []
    labels = []
    with open(input_file) as f:
        words = []
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[:2] == '1 ' and len(statements) > 0: # new article
                words = []
            if '\t' in line:
                question_parts = line.split('\t')
                tokens = re.sub(r'([\.\?])$', r' \1', question_parts[0].strip()).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

                dataset.append(words)
                labels.append(word_to_id[question_parts[1]])
            else:
                tokens = re.sub(r'([\.\?])$', r' \1', line).split()
                if update_word_ids:
                    for token in tokens[1:]:
                        if token not in word_to_id:
                            word_to_id[token] = word_id
                            word_id += 1

                for token in tokens[1:]:
                    words.append(word_to_id[token])

    return dataset, labels, word_to_id
