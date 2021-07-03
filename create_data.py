import os
import json
import re
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from collections import defaultdict
import tqdm
import string

if not os.path.exists('data.json'):
    qaps = defaultdict(list)
    with open('qaps.csv') as infile:
        lines = infile.read().splitlines()[1:]
        for idx, line in enumerate(lines):
            story_id = line.split(',')[0]
            q = line.split(',')[2]
            q = q.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            q = q.strip()
            a1 = line.split(',')[3]
            a2 = line.split(',')[4]
            a1 = a1.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            a1 = a1.strip()
            a2 = a2.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            a2 = a2.strip()
            qaps[story_id].append((q, a1, a2))

    id2type = {}
    with open('trunks/dataset_splits.json') as infile:
        dataset_splits = json.load(infile)
        for t in ['train', 'val', 'test']:
            for data in dataset_splits[t]:
                if data['type'] == 'book':
                    id2type[data['document_id']] = t

    all_data = defaultdict(list)

    for filename in tqdm.tqdm(os.listdir('trunks/books/paragraph200')):
        filename = os.path.join('trunks/books/paragraph200', filename)
        story_id = filename.split('.')[0].split('/')[-1]
        with open(filename) as story:
            corpus = [para.strip() for para in story]
            tokenized_corpus = [para.split() for para in corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        for qap in qaps[story_id]:
            q = qap[0]
            a1 = qap[1]
            a2 = qap[2]
            tokenized_query = q.split()
            r = bm25.get_top_n(tokenized_query, corpus, n=3)
            one_data = {'Q':q, 'A1':a1, 'A2':a2, 'R':r}
            all_data[id2type[story_id]].append(one_data)

    with open('data.json', 'w') as outfile:
        json.dump(all_data, outfile, indent=4)
else:
    with open('data.json') as infile:
        all_data = json.load(infile)
        os.makedirs('data', exist_ok=True)
        for t in ['train', 'val', 'test']:
            data_split = all_data[t]
            with open('data/{}.source'.format(t), 'w') as source, open('data/{}.target'.format(t), 'w') as target:
                for data in data_split:
                    source.write('{} <SEP> {}\n'.format(data['Q'], ' '.join(data['R'])))
                    source.write('{} <SEP> {}\n'.format(data['Q'], ' '.join(data['R'])))
                    target.write('{}\n{}\n'.format(data['A1'], data['A2']))
