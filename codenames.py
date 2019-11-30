from gensim import models
from random import randint
from itertools import combinations
import math

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
GOOD_COEF = 4
MEH_COEF = -1
BAD_COEF = -3
WORST_COEF = -5
RELEVANCE_THRESHOLD_LOW = 0.32
RELEVANCE_THRESHOLD_HIGH = 0.95

# ALGO 2 VARIABLES
MAX_CLUES = 5
MIN_CLUES = 2
SIM_FACTOR = (9/10)

def loadWords():
    wordsfile = open('words.txt', 'r')
    for line in wordsfile:
        word = line.strip()
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
    MODEL = models.KeyedVectors.load('best_500.bin', mmap='r')

def getWordSimilarity(word1, word2):
    return MODEL.similarity(word1, word2)

def getMove(team):
    if MODEL is None:
        return None
    goodWords = BOARD[team]
    mehWords = BOARD['tan']
    badWords = set()
    worstWords = BOARD['black']
    for k in BOARD:
        if not k in [team, 'tan', 'black']:
            badWords = badWords.union(BOARD[k])

    equation = []
    for word in goodWords:
        equation.append([GOOD_COEF, word])
    for word in mehWords:
        equation.append([MEH_COEF, word])
    for word in badWords:
        equation.append([BAD_COEF, word])
    for word in worstWords:
        equation.append([WORST_COEF, word])

    bestSoFar = [0, '', []]
    for vocabWord in MODEL.vocab.keys():
        sum = 0
        relevant = []
        for term in equation:
            distance = getWordSimilarity(vocabWord, term[1])
            if distance > RELEVANCE_THRESHOLD_LOW and distance < RELEVANCE_THRESHOLD_HIGH:
                sum += distance * term[0]
                relevant.append([term[1], distance, term[0]])
        if sum > bestSoFar[0]:
            bestSoFar = [sum, vocabWord, relevant]
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
        print bestCombo
    return bestCombo

def prettyPrintBoard():
    allWords = getAllWords()
    sq = math.floor(math.sqrt(NUM_BOARD_WORDS))
    c = 0
    for word in allWords:
        if c == sq:
            print ''
            c = 0
        print word, " " * (13 - len(word)),
        c +=1
    print ''
    for k in BOARD:
        print "---", str(k), "---"
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
