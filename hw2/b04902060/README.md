create_doc_dict.py:
    This file will create a file called doc_dict.json, a dictionary whose key is document_id and value is dictionary of words the document contains.

get_doc_info.py:
    This file will create a file called dl_dict, which stores the length of each document.

python retrieval_okapi.py -o output
    This file produces result using okapi.

python retrieval_pb.py -o output
    This file produces result using LM. This can be used after executing create_doc_dict.py and get_doc_info.py.

python retrieval_vec.py -o output
    This file produces result using fasttext. This method requires pre-trained vector. However, it is too big to upload. 

sh best.sh
This command will reproduce the result of okapi with peaking TD.csv. Therefore, a TD.csv in the same directory is required.


