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

with open('doc_dict.json') as f:
	doc_dict = json.load(f)

with open('dl_dict') as f:
    dl_dict = json.load(f)

REF_TOTAL = 0
for doc, length in dl_dict.items():
    REF_TOTAL += length

# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
num_corpus = corpus.shape[0] # used for random sample

MU = 125
LAMBDA = 0.3

# process each query
final_ans = []
for (query_id, query) in querys:
    print("query_id: {}".format(query_id))
	
	# counting query term frequency
    query_cnt = Counter()
    query_words = list(jieba.cut(query))
    query_cnt.update(query_words)

	# calculate scores by tf-idf
    document_scores = dict() # record candidate document and its scores
    for (word, count) in query_cnt.items():
        if word in invert_file:
            for doc, token_tf_dict in doc_dict.items():
                document_scores[doc] = 1
                doc_len = dl_dict[doc]
                c = 0
                prob_REF = invert_file[word]['idf']/REF_TOTAL
                if word in token_tf_dict:
                    c = token_tf_dict[word]
                prob = (c/(doc_len+MU)) + MU/(doc_len+MU) * prob_REF
                #prob = (1-LAMBDA)*c/doc_len + LAMBDA*prob_REF
                for i in range(count):
                    document_scores[doc] *= prob

	
	# sort the document score pair by the score
    sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
	
	# record the answer of this query to final_ans
    if len(sorted_document_scores) >= 300:
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300]])
    else: # if candidate documents less than 300, random sample some documents that are not in candidate list
        documents_set  = set([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
        sample_pool = ['news_%06d'%news_id for news_id in range(1, num_corpus+1) if 'news_%06d'%news_id not in documents_set]
        sample_ans = random.sample(sample_pool, 300-count)
        sorted_document_scores.extend(sample_ans)
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
	
# write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
    writer.writerow(head)
    for query_id, ans in enumerate(final_ans, 1):
        writer.writerow(['q_%02d'%query_id]+ans)
