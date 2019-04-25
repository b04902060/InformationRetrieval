import os
import argparse
from Vocabs import Vocabs
from DocVec import DocVec
from pathlib import Path
import xml.etree.cElementTree as ET
from tqdm import tqdm
import json
import math
import pandas as pd


DOCS_NUM = 46972
filename_dict = {}
reverse_filename_dict = {}



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', nargs='?', const=True, default=False, help='Turn on the relevance feedback.')
    parser.add_argument('-b', nargs='?', const=True, default=False, help='Use the best version.')
    parser.add_argument('-i', nargs='?', help='query-file')
    parser.add_argument('-o', nargs='?', help='ranked-list')
    parser.add_argument('-m', nargs='?', help='model-dir')
    parser.add_argument('-d', nargs='?', help='NTCIR-dir')
    parser.add_argument('-t', nargs='?', const=True, default=True, help='Turn on to test. (Otherwise validate)')
    parser.add_argument('-g', nargs='?', const=False, default=True, help='Turn on not to generate csv.')
    return parser.parse_args()


def preprocess(arg, useJSON=False, output_path='./model', createJSON=False, TEST=True):
    vocab_filename='vocabs_dic.json'
    docs_filename='docs_dic.json'
    inverted_filename = 'inverted-file'
    if TEST:
        vocab_filename = 'test-'+vocab_filename
        docs_filename = 'test-'+docs_filename
        inverted_filename = 'test-'+inverted_filename

    if(useJSON):
        with open(Path(output_path) / vocab_filename) as f:  
            vocabs_dic = json.load(f)
        with open(Path(output_path) / docs_filename) as f:  
            docs_dic = json.load(f)   
        return vocabs_dic, docs_dic

    else:
        vocabs_dic = {}
        docs_dic = []
        docs_len = []
        inverted_file_path = Path(arg.m) / inverted_filename
        for i in range(46972):
            docs_dic.append([])
            docs_len.append(0)
        
        with open(inverted_file_path) as f:
            for line in tqdm(f):
                vocab_key = line.split()[0]+' '+line.split()[1]
                c = int(line.split()[2])
                vocabs_dic[vocab_key] = []

                for i in range(c):
                    doc_line = [int(j) for j in f.readline().split()]
                    vocabs_dic[vocab_key].append(doc_line)
                    docs_dic[doc_line[0]].append([(int(line.split()[0]), int(line.split()[1])), doc_line[1]])
                    docs_len[doc_line[0]] += doc_line[1]
        if createJSON:
            with open(Path(output_path) / vocab_filename, 'w') as f:
                json.dump(vocabs_dic, f)
            with open(Path(output_path) / docs_filename, 'w') as f:
                json.dump(docs_dic, f)
        return vocabs_dic, docs_dic, docs_len

# def tf(t, d):
    # t: 


def queryPrepro(arg, vocabs_dic):
    if(arg.t): # testing
        tree = ET.ElementTree(file='./queries/query-test.xml')
    else: # validating
        tree = ET.ElementTree(file=arg.i)
    query = [] # [["1 2", "34 -1"], ["2 4"]]
    query_text = []
    vocab = {}

    with open(Path(arg.m) / "vocab.all", 'r') as f:
        i = 0
        for line in f:
            vocab[line[:-1]] = i
            i+=1

    # query_text: parse to continuous character
    for elem in tree.iter(tag='concepts'):
        query_text.append("".join(elem.text[1:-2].split('、')))

    # query: parse to string of index by looking up vocab.all 
    query = []
    for q_text in query_text:
        tmp = []
        for voc in q_text:
            if voc in vocab:
                tmp.append(vocab[voc])
            else:
                print(voc)
                tmp.append(-1)
        query.append([str(i) for i in tmp])
    
    # query_res: check uni-gram and bi-gram 
    query_res = []
    for q in query:
        tmp = []
        for i in range(len(q)):
            if (q[i] == '-1'):
                continue
            if (q[i]+' -1' in  vocabs_dic):
                tmp.append((int(q[i]), -1))
            if i < len(q)-1:
                if (q[i]+' '+q[i+1] in vocabs_dic):
                    tmp.append((int(q[i]), int(q[i+1])))
        tmp.sort()
        tmp = [[t, 1] for t in tmp]
        query_res.append(tmp)
    return query_res

def compute_score(query, doc_id, docs_dic, vocabs_dic, doc_len, avdl):
    doc_vec = docs_dic[doc_id]
    i = 0
    j = 0
    i_limit = len(doc_vec)
    j_limit = len(query)
    score = 0

    while(True):
        if (i >= i_limit or j >= j_limit):
            break
        if (doc_vec[i][0] > query[j][0]):
            j+=1
        elif (doc_vec[i][0] < query[j][0]):
            i+=1
        else:
            score += TFIDF(query[j], doc_id, i, doc_vec, vocabs_dic, doc_len, avdl) # TF(query[j], doc_id) * IDF(query[j]) * doc_vec[i][1]
            i+=1
            j+=1
    return score

def TFIDF(t, d, index, doc_vec, vocabs_dic, doc_len, avdl):
    # t: (2, -1)
    # d: doc_id
    # index: the index of t in d
    k1 = 1.5
    b = 0.75
    ka = 500
    TF = doc_vec[index][1] # frequency that t occurs in d (like count)
    QTF = t[1]
    N = 46972
    DF = len(vocabs_dic[str(t[0][0]) +' '+ str(t[0][1])])
    DL = doc_len
    
    final = math.log((N-DF+0.5)/(DF+0.5)) * ((k1+1)*TF)/((k1*(1-b+b*DL/avdl))+TF) * ((ka+1)*QTF)/(ka+QTF)  
    return final

def set_filename_dict(arg):
    with open(Path(arg.m) / 'file-list') as f:
        i = 0
        for line in f:
            filename = (line.split('/')[-1]).lower().replace('\n', '')
            filename_dict[filename] = i
            reverse_filename_dict[i] = filename
            i+=1

def getAnswer(arg):
    res = []
    with open(Path('./queries/ans_train.csv')) as f:
        f.readline()
        for line in f:
            tmp = []
            ans_doc = line.split(',')[-1].split(' ')
            tmp = [filename_dict[ans.replace('\n', '')] for ans in ans_doc]
            res.append(tmp)
    return res

def getPrecision(ans_list, pre_list):
    res = []
    for i in range(len(ans_list)):
        score = 0
        recall = 0
        for j in range(len(pre_list[i])):
            for k in range(len(ans_list[i])):
                if (pre_list[i][j] == ans_list[i][k]):
                    recall+=1
                    score+=(recall/(j+1))
        score /= len(ans_list[i])
        res.append(score)
    return res

def generateResult(arg, pre_list):
    ans_dict = {}
    ans_dict['query_id'] = ['0'+str(i+11) for i in range(len(pre_list))]
    pre_filename = []
    for p_list in pre_list:
        pre_filename.append(' '.join([reverse_filename_dict[i] for i in p_list]))
    ans_dict['retrieved_docs'] = pre_filename

    res = pd.DataFrame(ans_dict)
    if(arg.g):
        res.to_csv('rocchio_B_submission.csv', index=False)

    return res

def vecAdd(va, vb):
    res = []
    i = 0
    j = 0
    i_limit = len(va)
    j_limit = len(vb)
    while(True):
        if(i >= i_limit and j >= j_limit):
            break
        if(i >= i_limit):
            res.append(vb[j])
            j+=1
        elif(j >= j_limit):
            res.append(va[i])
            i+=2
        elif(va[i][0] > vb[j][0]):
            res.append(vb[j])
            j+=1
        elif(va[i][0] < vb[j][0]):
            res.append(va[i])
            i+=1
        else:
            res.append([va[i][0], va[i][1]+vb[j][1]])
            i+=1
            j+=1
    return res

def Rocchio(query, relevant_docs_id, irrelevant_docs_id, docs_dic, P):
    feedback_num = (len(relevant_docs_id), len(irrelevant_docs_id))
    res = [ [q[0], q[1]*P[0]] for q in query]

    # score tuple -> index
    relevant_docs_id = [r[1] for r in relevant_docs_id]
    irrelevant_docs_id = [r[1] for r in irrelevant_docs_id]

    Dr = []
    for id in relevant_docs_id: # relevant
        Dr = vecAdd(Dr, docs_dic[id])
    Dr = [ [d[0], d[1]*P[1]/feedback_num[0]]  for d in Dr]
    res = vecAdd(res, Dr)

    Dir = []
    for id in irrelevant_docs_id: # irrelevant
        Dir = vecAdd(Dir, docs_dic[id])
    Dr = [ [d[0], -d[1]*P[2]/feedback_num[1]]  for d in Dir]
    res = vecAdd(res, Dir)

    return res



def main():
    arg = args()
    set_filename_dict(arg)
    

    vocabs_dic, docs_dic, docs_len = preprocess(arg, useJSON=False, createJSON=False, TEST=False)
    avdl = sum(docs_len)/len(docs_len)

    query = queryPrepro(arg, vocabs_dic)

    pre_list = []
    feedback_time = 0
    if(arg.r):
        feedback_time = 2
        
    feedback_num = (10, 0)
    PARAMETERS = (0.8, 0.2, 0)
    converge_num1 = 2500
    converge_num2 = 1000

    for qu in query:
        q = qu
        candidate = []
        score_list = []
        for t in range(feedback_time+1):
            # converge the candidate after each feedback
            if (t==0):
                candidate = [i for i in range(46972)]
            else:
                if(t==1):
                    candidate = [s[1] for s in score_list[:converge_num1]]
                else:
                    candidate = [s[1] for s in score_list[:converge_num2]]

            score_list = []
            for id in tqdm(candidate):
                score = compute_score(q, id, docs_dic, vocabs_dic, docs_len[id], avdl)
                score_list.append((score, id))
            score_list.sort(reverse=True)
            if t < feedback_time:
                if(feedback_num[1] == 0):
                    q = Rocchio(q, score_list[:feedback_num[0]], [], docs_dic, PARAMETERS)
                else:
                    q = Rocchio(q, score_list[:feedback_num[0]], score_list[-feedback_num[1]:], docs_dic, PARAMETERS)
        pre_list.append([s[1] for s in score_list[:100]])

    
    if (arg.t):
        # generate result of test data
        res = generateResult(arg, pre_list)
    else:
        # validation
        ans_list = getAnswer(arg)
        final_score = getPrecision(ans_list, pre_list)
        print('final_each: ', final_score)
        print('final_avg:', sum(final_score)/len(final_score))

    

if __name__ == '__main__':
    main()