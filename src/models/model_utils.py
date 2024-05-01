# encoding=utf-8
from tqdm import tqdm
import csv
import numpy as np


def build_glove_embedding_dic(emb_path, vocab_path):
    # with open(emb_path, mode='r', encoding="utf-8") as f:
    #     embeddings = {}
    #     lines = f.readlines()
    #     pbar = tqdm(total=len(lines), desc='loading embeddings')
    #     for i, line in enumerate(lines):
    #         cols = line.split()
    #         embeddings[cols[0]] = [float(e) for e in cols[1:]]
    #         pbar.update(1)
    #     embeddings['unk_token'] = [float(0) for i in range(300)]
    # embeddings = {}
    # for i in range(400001):
    #     embeddings[i] = [float(0) for i in range(300)]
    vocab = {}
    with open(vocab_path, mode='r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            vocab[row[0]] = int(row[1])
    VOCAB_SIZE = len(vocab) + 1
    embeddings = np.zeros((VOCAB_SIZE, 300), dtype=np.float32)
    with open(emb_path, mode='r', encoding="utf-8") as f:
        for line in f:
            cols = line.split()
            idx = vocab.get(cols[0], 0)
            if idx > 0:
                for i in range(300):
                    embeddings[idx, i] = float(cols[i + 1])
    return vocab, embeddings


def kernel_mu(n_kernels):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu
    """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)


if __name__ == '__main__':
    print(kernel_mu(11))
    print(kernel_sigma(11))
