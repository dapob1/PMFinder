__author__ = 'oladapobakare'

import logging
import pprint
import sys

from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Clean and filter then create new file used in lda_1_0_2.py
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import itertools

tokenizer = RegexpTokenizer(r'\w+')  # use the Regexp to ignore non-alpha numeric characters

cachedStopWords = stopwords.words("english")  # Use stopwords from nltk


# Function to convert to lowercase, convert from raw file output, split & ignore non-alpha characters (via tokenize)
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
                if not str(strings).isalpha():
                    strings = ''
                bagofwords_row.append(strings)
        bagofwords.append(bagofwords_row)
    return bagofwords


dictionary = Dictionary.load("lsi_test_dict.dict")  # load a dictionary

# Open resume
with open("input_resume.txt", "rb") as inputFile:
    sampleResume = inputFile.read().splitlines()

# Clean resume
sampleResume = cleanConvert(sampleResume)
sampleResume = list(itertools.chain(*sampleResume))
sampleResume = filter(None, sampleResume)

# Load background corpus
# bow_corpus = MmCorpus("background_corpus.mm")

# Convert resume
bow_sampleResume = [dictionary.doc2bow(sampleResume) for document in sampleResume]
tfidf_transformation = TfidfModel.load("tfidf.model")

tfidf_sampleResume = tfidf_transformation[bow_sampleResume]

# lsi_transformation = LsiModel(corpus=tfidf_transformation[bow_corpus], id2word=dictionary, num_topics=400)
# lsi_transformation.save("lsi.model")
lsi_transformation = LsiModel.load("lsi.model")

from gensim.similarities import Similarity

# index_documents = texts

# corpus = [dictionary.doc2bow(text) for text in texts]

# index = Similarity(corpus=lsi_transformation[tfidf_transformation[corpus]], num_features=400, output_prefix="shard")

# index.save("similarity_index")
index = Similarity.load("similarity_index")

sims_to_query = index[lsi_transformation[tfidf_sampleResume]]

best_score = max(sims_to_query)

pprint.pprint(best_score)

print best_score[0]
