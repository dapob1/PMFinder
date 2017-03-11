__author__ = 'oladapobakare'

import itertools
import logging

import enchant
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# open training file
with open("PM_test_set_clean.csv", 'rb') as training_data:
    documents = training_data.read().splitlines()

with open("output.txt", 'rb') as training_data2:
    documents2 = training_data2.read().splitlines()

tokenizer = RegexpTokenizer(r'\w+')  # use the Regexp to ignore non-alpha numeric characters

cachedStopWords = stopwords.words("english")
isUSenglish = enchant.Dict('en_US')
isGBenglish = enchant.Dict('en_GB')


# Function to keep only english (US and GB) alpha characters
def cleanEnglish(input_file):
    myCleanEnglish = []
    for doc in input_file:
        for strings in doc:
            if not str(strings).isalpha():
                doc = ''
            elif isUSenglish.check(strings) == False:
                if isGBenglish.check(strings) == False:
                    doc = ''
            myCleanEnglish.append(strings)
    return myCleanEnglish


# Function to loop through to convert to lowercase, convert from raw file output, split & ignore non-alpha characters (via tokenize)
def cleanConvert(input_file):
    bagofwords = []
    for doc in input_file:
        doc = doc.lower()
        doc = tokenizer.tokenize(doc)
        bagofwords_row = []
        for strings in doc:
            if strings not in cachedStopWords:
                strings = strings.decode('latin-1').encode(
                    "utf-8")  # convert from latin 1 output from salestools.io scraper to utf-8

                # if not str(strings).isalpha():
                # strings = ''
                # elif isUSenglish.check(strings) == False:
                #   if isGBenglish.check(strings) == False:
                #      strings = ''


                bagofwords_row.append(strings)
        bagofwords.append(bagofwords_row)
    return bagofwords


final_test = []
documents = cleanEnglish(documents)
texts = cleanConvert(documents)

print len(texts)
texts = list(itertools.chain(*texts))

print len(texts)
blank = []
blank.append(texts)
texts = blank

texts = filter(None, texts)  # remove empty spaces in the docs
print len(texts)
# final_test.append()


"""
# Creating a dictionary to represent each document by an id)
dictionary = corpora.Dictionary(texts)
# Remove extremes
dictionary.filter_extremes(no_below=0, no_above=1, keep_n=100000)

dictionary.save('PM_test_set.dict')
print (dictionary.token2id)

import operator
sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
print sorted_dictionary

#dictionary.load('PM_test_set.dict')

# To convert documents to vectors
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('PM_test_set.mm', corpus)


# Transform to tfidf (term frequency - inverse document frequency
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# LDA transformation
#lda = models.LdaModel(corpus, id2word=dictionary, num_topics=30, passes=100 )

# LSI transformation
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)

# set or load up similarities for LDA
#index = similarities.MatrixSimilarity(lda[corpus])

# set or load up similarities for LDA
index = similarities.MatrixSimilarity(lsi[corpus])

#index.save("simIndex.index")
#index = similarities.MatrixSimilarity.load("simIndex.index")
#index = similarities.Similarity('test_output.txt',corpus, num_features=12, num_best=3)

"""
"""
# Function to read input file and split lines (do I need to though)
def readInputFile(file):
    with open(file,'rb') as inputFile:
        return inputFile.read().splitlines()
"""
"""
# open training file
with open("input_zero.txt", 'rb') as resume:
    member_resume = resume.read().splitlines()

input_resume = cleanConvert(member_resume)

print input_resume
#member_resume = "Oladapo"
input_resume = list(itertools.chain(*input_resume))
blank =[]
blank.append(input_resume)
input_resume = blank
input_resume = filter(None, input_resume)
print input_resume


vec_bow = [dictionary.doc2bow(rez) for rez in input_resume]
#print vec_bow
#vec_lda = lda[vec_bow]

vec_lsi = lsi[vec_bow]
#print vec_lsi
#sims = index[vec_lda]
sims = index[vec_lsi]

#print (list(enumerate(sims)))
sims = sorted(enumerate(sims), key=lambda item: -item[1])
#print len(filter(lambda x: any(str(x)) >= 0.8, sims[0]))
#print all(sims[0] > 0.8 for sim in sims)
#print sum(int(x) for x in sims)

print sims
"""
"""
# remove common word
"""
"""
stoplist = set('for a of the and to in'.split())

texts = [[word for word in document.split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
#texts = documents
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint   # pretty-printer
pprint(texts)

"""
"""Hidden models to move to another file. Maybe create an option to select transformation method"""
"""tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]
#for doc in corpus_tfidf:
    #print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]

lsi.print_topics(2)

for doc in corpus_lsi:
    print(doc)




lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lda = lda[corpus_tfidf]



mallet_path = 'mallet-2.0.8RC2/bin/mallet'

lda = gensim.models.wrappers.LdaMallet(mallet_path, corpus, num_topics=10,id2word=dictionary) # LdaMallet was moved to wrappers
corpus_lda = lda[corpus_tfidf]

for doc in corpus_lda:
    print(doc)
"""
"""
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]

for doc in corpus_lsi:
    print(doc)

doc = "Formulated marketing strategy for launching Login with Amazon"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print vec_lsi

index = similarities.MatrixSimilarity(lsi[corpus_tfidf]) # transform corpus to LSI space and index it

index.save('/test_resume_lsi.index') # Index persistency
index = similarities.MatrixSimilarity.load('/test_resume_lsi.index')

sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1]) # perform a similarity query against the corpus
print (sims) # print sorted (document number, similarity score) 2-tuples

#model = gensim.models.wrappers.LdaMallet(mallet_path, corpus, num_topics=10,id2word=corpus.dictionary) # LdaMallet was moved to wrappers

# will need to ensure corpus persistency by using different corpus formats See https://radimrehurek.com/gensim/tut1.html

#for vector in corpus:
    #print(vector)
"""
"""
mallet_path = 'mallet-2.0.8RC2/bin/mallet'

model = gensim.models.wrappers.LdaMallet(mallet_path, corpus, num_topics=10,id2word=corpus.dictionary) # LdaMallet was moved to wrappers

print model[corpus]
"""

# Old code using https://github.com/ariddell/lda
"""import numpy as np
import lda
import lda.datasets
#from numpy import genfromtxt

test_data = np.genfromtxt('output.txt', delimiter=':')

X = np.array(test_data)
#X = lda.datasets.load_reuters()
#vocab = lda.datasets.load_reuters_vocab()
#titles = lda.datasets.load_reuters_titles()
#X.shape
#X.sum()

model = lda.LDA(n_topics=20, n_iter=100, random_state=0)

model.fit(X)

topic_word = model.components__

n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))"""

"""
# loop through to convert to lowercase, convert from raw file output, split & ignore non-alpha characters (via tokenize)
for doc in documents:
    doc = doc.lower()
    doc = tokenizer.tokenize(doc)
    texts_row = []
    for strings in doc:
        if strings not in cachedStopWords:
            strings = strings.decode('latin-1').encode("utf-8") # convert from latin 1 output from salestools.io scraper to utf-8
            texts_row.append(strings)
    texts.append(texts_row)
"""
