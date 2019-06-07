import json
import jieba
import pandas as pd
import numpy as np
import random
import csv
import operator
from argparse import ArgumentParser
from collections import Counter
from math import log

def find_k_nearest(source, vectors, k):
    norm1 = np.linalg.norm(source)
    norm2 = np.linalg.norm(vectors, axis=1)
    cosine_similarity = np.sum(source * vectors, axis=1) / norm1 / norm2
    return np.argsort(cosine_similarity)[::-1][1:(k + 1)]


parser = ArgumentParser()
parser.add_argument("-i", "--inverted_file", default='inverted_file.json', dest = "inverted_file", help = "Pass in a .json file.")
parser.add_argument("-q", "--query_file", default='QS_1.csv', dest = "query_file", help = "Pass in a .csv file.")
parser.add_argument("-c", "--corpus_file", default='NC_1.csv', dest = "corpus_file", help = "Pass in a .csv file.")
parser.add_argument("-o", "--output_file", default='sample_output.csv', dest = "output_file", help = "Pass in a .csv file.")

args = parser.parse_args()

# load inverted file
with open(args.inverted_file) as f:
    invert_file = json.load(f)

# avdl and dl
dl = {}
with open('dl_dict') as f:
    dl = json.load(f)
avdl = 195.69838520988586

# number of total documents
N = len(dl)

# hyperparameter
k1 = 1
b = 0.75
ka = 150


# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
num_corpus = corpus.shape[0] # used for random sample

q = [['通姦', '在', '刑法', '上','應該', '除罪', '化'],
['應該', '取消', '機車', '強制', '二段式', '左轉', '(', '待轉', ')'],
['支持', '博弈', '特區', '在', '台灣', '合法化'],
['中華', '航空', '空服員', '罷工', '是', '合理', '的'],
['性交易', '應該', '合法化'],
['ECFA', '早收','清單','可', '（', '有', '）', '達到', '其', '預期', '成效'],
['應該', '減免', '證所稅'],
['贊成', '中油', '在', '觀塘', '興建', '第三', '天然','氣', '接收站'],
['支持', '中國','學生', '納入', '健保'],
['支持', '台灣', '中', '小','學', '（', '含','高職', '、', '專科', '）', '服儀', '規定', '（', '含', '髮', '、', '襪', '、', '鞋', '）', '給予', '學生', '自主'],
['不', '支持', '使用', '加密', '貨幣'],
['不', '支持', '學雜費', '調漲'],
['同意', '政府', '舉債', '發展', '前瞻', '建設', '計畫'],
['支持', '電競', '列入', '體育', '競技'],
['反對', '台鐵', '東移', '徵收', '案'],
['支持', '陳', '前', '總統', '保外', '就醫'],
['年金', '改革', '應', '取消', '或應', '調降', '軍','公','教', '月', '退', '之', '優存', '利率', '十八', '趴'],
['同意', '動物', '實驗'],
['油價', '應該', '凍漲', '或', '緩漲'],
['反對', '旺旺', '中時', '併','購','中','嘉']]

with open('docs_index_dict.json') as f:
    docs_index_dict = json.load(f)
docs_vec = np.load('docs_vec.npy')

words_dict = {}
vectors = []
print('Loading word vector...')
with open('../zh_wiki_fasttext_300.txt') as f:
    # skip first line
    f.readline()
    line = f.readline()
    i = 0
    while len(line) > 0:
        line = line.split(' ')
        words_dict[line[0]] = i
        vectors.append(np.array([float(x) for x in line[1:]]))
        line = f.readline()
        i += 1
vectors = np.vstack(vectors)

def vec_score(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# process each query
final_ans = []
i = 0
for (query_id, query) in querys:
    print("query_id: {}".format(query_id))
    
    # counting query term frequency
    query_cnt = Counter()
    query_words = list(jieba.cut(query))
    query_cnt.update(query_words)
    #print(query_words)

    # calculate scores by tf-idf
    document_scores = dict() # record candidate document and its scores
    #for (word, count) in query_cnt.items():
    query_vec = np.zeros(300)
    for word in q[i]:
        if word in words_dict:
            query_vec += vectors[words_dict[word]]
    for doc, doc_index in docs_index_dict.items():
        doc_vec = docs_vec[doc_index]
        document_scores[doc] = vec_score(query_vec, doc_vec)
    for word in q[i]:
        if word in words_dict:
            query_vec = vectors[words_dict[word]]
        

    i += 1
    # sort the document score pair by the score
    sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    # record the answer of this query to final_ans
    if len(sorted_document_scores) >= 300:
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300]])
    else: # if candidate documents less than 300, random sample some documents that are not in candidate list
        documents_set  = set([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
        sample_pool = ['news_%06d'%news_id for news_id in range(1, num_corpus+1) if 'news_%06d'%news_id not in documents_set]
        sample_ans = random.sample(sample_pool, 300-len(sorted_document_scores))
        sorted_document_scores.extend(sample_ans)
        final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
    
# write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
    writer.writerow(head)
    for query_id, ans in enumerate(final_ans, 1):
        writer.writerow(['q_%02d'%query_id]+ans)
