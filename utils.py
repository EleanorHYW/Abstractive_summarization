import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import gensim
import torch
import word2vec  # used for load word2vec embeddings, comment this line if you don't need this
import os
import config

pad_index = 0

""" Caution:
In training data, unk_tok='<unk>', but in test data, unk_tok='UNK'.
This is reasonable, because if the unk_tok you predict is the same as the
unk_tok in the test data, then your prediction would be regard as correct,
but since unk_tok is unknown, it's impossible to give a correct prediction
"""


def my_pad_sequence(batch, pad_tok):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_tok] * (max_len - len(b)) for b in batch]
    return batch


class BatchManager:
    def __init__(self, data, batch_size, vocab):
        self.steps = int(len(data) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(data):
            self.steps += 1
        self.vocab = vocab
        self.data = data
        self.batch_size = batch_size
        self.bid = 0

    def next_batch(self, pad_flag=True, cuda=True):
        stncs = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        if pad_flag:
            stncs = my_pad_sequence(stncs, config.pad_tok)
            ids = [[self.vocab.get(tok, self.vocab[config.unk_tok]) for tok in stnc] for stnc in stncs]
            ids = torch.tensor(ids)
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return stncs, ids.cuda() if cuda else ids


def build_vocab(filelist, vocab_file='sumdata/vocab.json', min_count=0, vocab_size=1e9):
    print("Building vocab with min_count=%d..." % min_count)
    word_freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                word_freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(word_freq))

    if config.unk_tok in word_freq:
        word_freq.pop(config.unk_tok)
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    vocab = {config.pad_tok: 0, config.start_tok: 1, config.end_tok: 2, config.unk_tok: 3}
    for word, freq in sorted_freq:
        if freq > min_count:
            vocab[word] = len(vocab)
        if len(vocab) == vocab_size:
            break
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab)/len(word_freq)*100))

    json.dump(vocab, open(vocab_file,'w'))
    return vocab


def load_embedding_vocab(embedding_path):
    fin = open(embedding_path)
    vocab = set([])
    for _, line in enumerate(fin):
        vocab.add(line.split()[0])
    return vocab


def load_word2vec_embedding(filepath):
    # f = 'bert-base-multilingual-cased.119547.768d.vec'
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=False)
    vocabulary = model.vocab
    vocab = {}
    weights = []
    for word in vocabulary:
        vocab[word] = len(vocab)
        vec = model[word]
        weights.append(vec)
    if not vocab.__contains__(config.start_tok):
        vocab[config.start_tok] = vocab["[unused1]"]
        vocab.pop("[unused1]")
    if not vocab.__contains__(config.end_tok):
        vocab[config.end_tok] = vocab["[unused2]"]
        vocab.pop("[unused2]")
    if not vocab.__contains__(config.end_tok):
        vocab[config.unk_tok] = vocab["[unused3]"]
        vocab.pop("[unused3]")
    if not vocab.__contains__(config.pad_tok):
        vocab[config.end_tok] = vocab["[unused4]"]
        vocab.pop("[unused4]")
    json_vocab = json.dumps(vocab)
    with open('sumdata/bert_vocab_large_uncased.txt', 'w') as json_file:
        json_file.write(json_vocab)

    return vocab, torch.tensor(weights, dtype=torch.float)


def get_vocab(TRAIN_X, TRAIN_Y):
    src_vocab_file = "sumdata/src_vocab.json"
    if not os.path.exists(src_vocab_file):
        src_vocab = build_vocab([TRAIN_X], src_vocab_file)
    else:
        src_vocab = json.load(open(src_vocab_file))

    tgt_vocab_file = "sumdata/tgt_vocab.json"
    if not os.path.exists(tgt_vocab_file):
        tgt_vocab = build_vocab([TRAIN_Y], tgt_vocab_file)
    else:
        tgt_vocab = json.load(open(tgt_vocab_file))
    return src_vocab, tgt_vocab


def load_data(filename, max_len, n_data=None):
    """
    :param filename: the file to read
    :param max_len: maximum length of a line
    :param vocab: dict {word: id}, if no vocab provided, return raw text
    :param n_data: number of lines to read
    :return: datas
    """
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        if len(words) > max_len - 2:
            words = words[:max_len-2]
        words = [config.start_tok] + words + [config.end_tok]
        datas.append(words)
    return datas


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, n_data=None):
        self.datas = load_data(filename, vocab, n_data)
        self._size = len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return self._size


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """ GPU memory debugger
    Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)
