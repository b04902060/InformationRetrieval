import json


with open('inverted_file.json') as f:
    invert_file = json.load(f)

N = len(invert_file)
dl = {}

for word in invert_file:
    docs = invert_file[word]['docs']
    for doc_num in docs:
        for doc, num in doc_num.items():
            if doc in dl:
                dl[doc] += num
            else:
                dl[doc] = num

with open('dl_dict', 'w') as f:
    json.dump(dl, f)

total = 0
for doc, length in dl.items():
    total += length

avdl = total/N
print("avdl=", avdl)

