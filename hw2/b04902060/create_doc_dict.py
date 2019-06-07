import json
import jieba
import pandas as pd
import numpy as np
import random
import csv
import operator
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument("-i", "--inverted_file", default='inverted_file.json', dest = "inverted_file", help = "Pass in a .json file.")
parser.add_argument("-q", "--query_file", default='QS_1.csv', dest = "query_file", help = "Pass in a .csv file.")
parser.add_argument("-c", "--corpus_file", default='NC_1.csv', dest = "corpus_file", help = "Pass in a .csv file.")
parser.add_argument("-o", "--output_file", default='sample_output.csv', dest = "output_file", help = "Pass in a .csv file.")

args = parser.parse_args()

# load inverted file
with open(args.inverted_file) as f:
	invert_file = json.load(f)

# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
num_corpus = corpus.shape[0] # used for random sample

doc_dict = {}
for token, value in invert_file.items():
	docs = value['docs']
	for document_count_dict in docs:
		for doc, doc_tf in document_count_dict.items():
			if doc in doc_dict:
				if token in doc_dict[doc]:
					doc_dict[doc][token] += doc_tf
				else:
					doc_dict[doc][token] = doc_tf
			else:
				doc_dict[doc] = {}
				doc_dict[doc][token] = doc_tf

with open('doc_dict.json', 'w') as f:
	json.dump(doc_dict, f)

