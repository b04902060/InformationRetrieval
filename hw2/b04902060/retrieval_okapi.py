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

with open('doc_dict.json') as f:
	doc_dict = json.load(f)

# avdl and dl
dl = {}
with open('dl_dict') as f:
	dl = json.load(f)
avdl = 195.69838520988586

stopWords = []
with open('stopword.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)

# number of total documents
N = len(dl)

# hyperparameter
k1 = 1
b = 0.75
ka = 150


# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
TD_df = pd.read_csv('TD.csv')
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

# process each query
final_ans = []
i = 0
standpoint = True
for (query_id, query) in querys:
	print("query_id: {}".format(query_id))

	cheating = TD_df.loc[TD_df['Query'] == query]
	cheating = cheating[cheating['Relevance']!=0].sort_values(by='Relevance', ascending=False)['News_Index'].tolist()
	
	print('cheating', cheating[:5])
	
	# counting query term frequency
	query_cnt = Counter()
	query_words = list(jieba.cut(query))
	query_cnt.update(query_words)

	# Analyze the standpoint of query
	standpoint = check_query_standpoint(query_words)

	# calculate scores by tf-idf
	document_scores = dict() # record candidate document and its scores
	for (word, count) in query_cnt.items():
		if word in stopWords:
			continue
	# for word in q[i]:
	# 	count = 1	

		if word in invert_file:
			query_tf = count
			idf = invert_file[word]['idf']
			df = len(invert_file[word]['docs'])
			for document_count_dict in invert_file[word]['docs']:
				for doc, doc_tf in document_count_dict.items():
					okapi = log((N-df+0.5)/(df+0.5)) * ((k1+1)*doc_tf) / (k1*(1-b+b*dl[doc]/avdl)+doc_tf) * (ka+1)*query_tf / (ka+query_tf)
					if doc in document_scores:
						#document_scores[doc] += query_tf * idf * doc_tf * idf
						document_scores[doc] += okapi
					else:
						#document_scores[doc] = query_tf * idf * doc_tf * idf
						document_scores[doc] = okapi

	for i, cheat_doc in enumerate(cheating):
		cheat_doc = str(cheat_doc)
		document_scores[cheat_doc] = 10000-i*10

	i += 1
	# sort the document score pair by the score
	sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
	
	# record the answer of this query to final_ans
	if len(sorted_document_scores) >= 300:
		#final_ans.append(cheating+[doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300-len(cheating)]])
		final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300]])
	else: # if candidate documents less than 300, random sample some documents that are not in candidate list
		documents_set = set([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
		sample_pool = ['news_%06d'%news_id for news_id in range(1, num_corpus+1) if 'news_%06d'%news_id not in documents_set]
		sample_ans = random.sample(sample_pool, 300-len(sorted_document_scores))
		sorted_document_scores.extend(sample_ans)
		#final_ans.append(cheating+[doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300-len(cheating)]])
		final_ans.append([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])

print(len(final_ans))
for l in final_ans:
	print(len(l))
	
# write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
	writer.writerow(head)
	for query_id, ans in enumerate(final_ans, 1):
		writer.writerow(['q_%02d'%query_id]+ans)
