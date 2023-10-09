from typing import List, Dict, Any
import json
import pickle
import random
import tqdm
from sentence_transformers import InputExample


def read_jsonl(filepath) -> List[Dict[str, Any]]:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            sample = json.loads(row.strip())
            data[sample['_id']] = sample['text']
            if 'title' in sample and sample['title'] is not None and len(sample['text']) > 0:
                data[sample['_id']] = sample['title'].replace('_', " ") + " " + sample['text']
    return data

def read_tsv(filepath: str) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            data.append(row.strip().split('\t'))
    return data

def load_pickle(filepath: str):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def augment_queries(x, p_mask=0.1):
    return ' '.join([tok for tok in x.split() if random.random() > p_mask])


def load_data(datasets_root, dataset_name, p_mask):
    train_corpus = {}
    for pid, p in read_jsonl(f"{datasets_root}/{dataset_name}/train.corpus.jsonl").items():
        train_corpus[pid] = p
    print("train_corpus", len(train_corpus), next(iter(train_corpus.items())))

    train_queries = {}
    for qid, q in read_jsonl(f"{datasets_root}/{dataset_name}/train.queries.jsonl").items():
        train_queries[qid] = augment_queries(q, p_mask=p_mask)
    print("train_queries", len(train_queries), next(iter(train_queries.items())))

    train_triples = []
    train_triples.extend(read_tsv(f"{datasets_root}/{dataset_name}/train.triples.tsv"))

    return train_corpus, train_queries, train_triples


def create_train_dev_triples(train_triples):
    cutoff = len(train_triples) - (len(train_triples) // 100)
    random.shuffle(train_triples)
    dev_triples = train_triples[cutoff:]
    train_triples = train_triples[:cutoff]
    print("train_triples", len(train_triples), next(iter(train_triples)))
    print("dev_triples", len(dev_triples), next(iter(dev_triples)))

    return train_triples, dev_triples

def create_dev_data(dev_triples, train_queries, train_corpus, num_dev_queries, num_max_dev_negatives):
    dev_samples = {}

    ### DEV DATA
    for qid, pos_id, neg_id in dev_triples:

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': train_queries[qid], 'positive': set(), 'negative': set()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].add(train_corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(train_corpus[neg_id])

    print("dev: ", len(dev_samples))
    return dev_samples


def create_train_data(train_corpus, train_triples, dev_samples, train_queries, pos_neg_ration, max_train_samples):
    ### Now we create our training & dev data
    train_samples = []

    ### TRAIN DATA
    cnt = 0
    for qid, pos_id, neg_id in tqdm.tqdm(train_triples, unit_scale=True):

        if qid in dev_samples:
            continue

        if qid not in train_queries:
            print(f"{qid} not in train_queries")
            continue

        query = train_queries[qid]
        if (cnt % (pos_neg_ration+1)) == 0:
            passage = train_corpus[pos_id]
            label = 1
        else:
            passage = train_corpus[neg_id]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

    print("train: ", len(train_samples))
    return train_samples