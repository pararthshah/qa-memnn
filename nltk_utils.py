import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN

def memoize1(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def memoize2(f):
    memo = {}
    def helper(x,y):
        if (x,y) not in memo:
            memo[(x,y)] = f(x, y)
        return memo[(x,y)]
    return helper

def stem_word(word):
    return nltk.stem.snowball.EnglishStemmer().stem(word)

stem_word = memoize1(stem_word)

def get_lemma(word, tag):
    return WordNetLemmatizer().lemmatize(word, tag)

get_lemma = memoize2(get_lemma)

def canonicalize_tokens(tokens):
    canonical_tokens = []
    tags = nltk.pos_tag(tokens)
    for tag in tags:
        wn_tag = penn_to_wn(tag[1])
        t = get_lemma(tag[0], wn_tag)
        t = stem_word(t)
        canonical_tokens.append(t)
    return canonical_tokens
