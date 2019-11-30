from gensim import models
import gensim
from nltk import find
from nltk.corpus import brown
from nltk import FreqDist
from random import randint
from itertools import combinations
import more_itertools
import math
from math import inf
import numpy as np
from numpy import dot
from numpy.linalg import norm

# GAME DEFINITION
NUM_BOARD_WORDS = 25
NUM_PER_TEAM = 9
NUM_TEAMS = 2
NUM_BLACK = 1
WORDS = set()
BOARD = {}

# GENERAL MODEL VARIABLES
MODEL = None

# ALGO 1 VARIABLES
GOOD_COEF = 10
MEH_COEF = -1
BAD_COEF = -5
WORST_COEF = -10
RELEVANCE_THRESHOLD_LOW = 0.32
RELEVANCE_THRESHOLD_HIGH = 0.95

# ALGO 2 VARIABLES
MAX_CLUES = 5
MIN_CLUES = 2
SIM_FACTOR = (9/10)

def loadWords():
    wordsfile = open('words.txt', 'r')
    for line in wordsfile:
        word = line.strip().lower()
        if word in MODEL.vocab.keys():
            WORDS.add(word)

def getRandomWord(usedWords):
    diff = WORDS.difference(usedWords)
    if len(diff) == 0:
        raise Exception('All words used. Can\'t get new word.')
    word = list(diff)[randint(0, len(diff)-1)]
    return word

def fillBoard():
    usedWords = set()
    for i in range(NUM_TEAMS):
        BOARD[i] = set()
        for j in range(NUM_PER_TEAM):
            word = getRandomWord(usedWords)
            usedWords.add(word)
            BOARD[i].add(word)
    BOARD['black'] = set()
    for i in range(NUM_BLACK):
        word = getRandomWord(usedWords)
        usedWords.add(word)
        BOARD['black'].add(word)
    BOARD['tan'] = set()
    for i in range(NUM_BOARD_WORDS - (NUM_PER_TEAM * NUM_TEAMS) - NUM_BLACK):
        word = getRandomWord(usedWords)
        usedWords.add(word)
        BOARD['tan'].add(word)

def getAllWords():
    words = set()
    for k in BOARD:
        words = words.union(BOARD[k])
    return words

def loadModel(lim):
    global MODEL
    #MODEL = models.KeyedVectors.load('best_500.bin', mmap='r')
    #word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
    #MODEL = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    #MODEL = gensim.models.KeyedVectors.load('brownlow.txt', mmap='r')
    MODEL = gensim.models.KeyedVectors.load('gnews100000.txt', mmap='r')

def getWordSimilarity(word1, word2):
    return MODEL.similarity(word1, word2)

def cos_sim(a, b):
    return dot(a,b) / (norm(a) * norm(b))

def score(wset, vectors, mehWords, badWords, worstWords):
    center_of_mass = sum(vectors[w] for w in wset)
    center_of_mass /= norm(center_of_mass)

    score = 0
    for w in wset:
      score += cos_sim(vectors[w], center_of_mass) * GOOD_COEF

    for w in mehWords:
      score += cos_sim(vectors[w], center_of_mass) * MEH_COEF * len(wset)

    for w in badWords:
      score += cos_sim(vectors[w], center_of_mass) * BAD_COEF * len(wset)

    for w in worstWords:
      score += cos_sim(vectors[w], center_of_mass) * WORST_COEF * len(wset)

    return score, center_of_mass

def score_partition(partition, vectors, mehWords, badWords, worstWords):
    wset_to_sc_com = {}
    for wset in partition:
      wset_to_sc_com[tuple(wset)] = score(wset, vectors, mehWords, badWords, worstWords)

    total_score = 1.01**(((1/len(wset_to_sc_com))**1.2) * (sum(wset_to_sc_com[wset][0] * len(wset) for wset in wset_to_sc_com)))
    return total_score, wset_to_sc_com

def find_closest(com, candidate_words, vectors, allWords):
    return max({cw for cw in candidate_words if cw[0] not in allWords and
                all((cw[0] not in word) and (word not in cw[0]) for word in allWords)},
               key=lambda word_freq: cos_sim(com, vectors[word_freq[0]]))[0]

def getMove(team):
    if MODEL is None:
        return None

    dist = FreqDist(w.lower() for w in brown.words())
    allWords = getAllWords()

    candidate_words = dist.most_common(50000)
    goodWords = BOARD[team]
    #candidate_words = {cw for cw in candidate_words if (cw[0] in MODEL.vocab.keys())}
    candidate_words = {(word, 0) for word in MODEL.vocab.keys() if '_' not in word}
    mehWords = BOARD['tan']
    badWords = set()
    worstWords = BOARD['black']

    vectors = {word[0] : MODEL.get_vector(word[0]) for word in candidate_words}
    #print('vectorslen!!!!', len(vectors))

    for k in BOARD:
        if not k in [team, 'tan', 'black']:
            badWords = badWords.union(BOARD[k])

    partitions = more_itertools.set_partitions(goodWords)
    best_score = -float(inf)
    best_partition = {}
    partition_sizes = np.zeros(9)
    partition_scores = np.zeros(9)
    partition_scores_max = np.ones(9) * -float(inf)
    for partition in partitions:
        sc, partition_to_sc_com = score_partition(partition, vectors, mehWords, badWords, worstWords)
        ps = len(partition_to_sc_com)
        partition_sizes[ps-1] += 1
        partition_scores[ps-1] += sc
        partition_scores_max[ps-1] = max(sc, partition_scores_max[ps-1])

        if sc > best_score:
          best_score = sc
          best_partition = partition_to_sc_com

    print(partition_sizes)
    print(partition_scores_max)
    print(partition_scores/partition_sizes)

    #print(best_partition)
    print(len(best_partition), sum(len(wset) for wset in best_partition))
    first_wset = max(best_partition, key=lambda x: best_partition[x][0])
    com = best_partition[first_wset][1]

    target = find_closest(com, candidate_words, vectors, allWords)
    bestSoFar = [best_partition[first_wset][0], target, first_wset]
    return bestSoFar

def getMove2(team):
    if MODEL is None:
        return None
    bestCombo = [0, (), (), ()]
    allWords = getAllWords()
    for i in range(MIN_CLUES, MAX_CLUES+1):
        combos = combinations(BOARD[team], i)
        for combo in combos:
            combo = set(combo)
            badWords = combo.difference(allWords)
            results = MODEL.most_similar(positive=combo, negative=badWords, topn=10)
            for result in results:
                guess = result[0].capitalize()
                if guess in combo or (guess + 's') in combo or (guess[-1] == "s" and guess[-1] in combo):
                    continue
                sim = result[1] * math.pow(i, SIM_FACTOR)
                if sim > bestCombo[0]:
                    bestCombo = [sim, result[0], combo]
        print(bestCombo)
    return bestCombo

def prettyPrintBoard():
    allWords = getAllWords()
    sq = math.floor(math.sqrt(NUM_BOARD_WORDS))
    c = 0
    for word in allWords:
        if c == sq:
            print('')
            c = 0
        print(word, " " * (13 - len(word)))
        c +=1
    print('')
    for k in BOARD:
        print ("---", str(k), "---")
        for word in BOARD[k]:
            print(word)

loadModel(500000)
loadWords()
fillBoard()
prettyPrintBoard()
move1 = getMove(0)
print(move1)
move2 = getMove(1)
print(move2)
