# encoding=utf-8
import os
import codecs
import yaml
import datetime
import pickle

def loadyaml(path):
    '''
     Read the config file with yaml
    :param path: the config file path
    :return: bidict
    '''
    doc = []
    if os.path.exists(path):
        with codecs.open(path, 'r') as yf:
            doc = yaml.safe_load(yf)
    return doc


def calculate_mrr(rels, retrieved):
    mrr = 0
    for qid, docs in retrieved.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        for i in range(min(len(ranked), 100)):
            if ranked[i] in rels[qid]:
                mrr += 1 / (i + 1)
    mrr /= len(retrieved)
    return mrr


def print_message(*s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), *s), flush=True)


if __name__ == "__main__":
    pass