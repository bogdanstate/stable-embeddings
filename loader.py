import torch
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from config import MIN_COUNT, WINDOW_SIZE, BATCH_SIZE


def get_loader(corpus, timestamp):
    words = Counter(corpus.words())
    thresholded = [k for k, v in words.items() if v >= MIN_COUNT]
    unk_idx = len(thresholded)
    start_idx = unk_idx + 1
    end_idx = start_idx + 1

    with open('words_lookup_%d.txt' % timestamp, 'w') as f:
        for word, idx in zip(thresholded, range(len(thresholded))):
            f.write("%s\t%d\n" % (word, idx))
        f.write("UNK\t%d\n" % unk_idx)
        f.write("START\t%d\n" % start_idx)
        f.write("END\t%d\n" % end_idx)

    sents = [
        [thresholded.index(y)
         if y in thresholded else unk_idx
         for y in x]
        for x in corpus.sents()
    ]
    words = dict((k, v) for k, v in zip(range(len(thresholded)), thresholded))
    words[unk_idx] = 'UNK'
    words[start_idx] = 'START'
    words[end_idx] = 'END'

    sents = [
        [start_idx] * WINDOW_SIZE + x + [end_idx] * WINDOW_SIZE
        for x in sents
    ]
    num_entities = end_idx + 1
    windows = [
        x[(k - WINDOW_SIZE):(k + WINDOW_SIZE + 1)]
        for x in sents
        for k in range(WINDOW_SIZE, len(x) - WINDOW_SIZE - 1)
    ]
    feature_range = ([x for x in range(0, WINDOW_SIZE)] +
                     [x for x in range(WINDOW_SIZE + 1, 2 *
                                       WINDOW_SIZE + 1)])
    focal_ids = [x[WINDOW_SIZE] for x in windows]
    features_ids = [
        [y[i] for i in feature_range] for y in windows
    ]

    dataset = TensorDataset(torch.LongTensor(focal_ids),
                            torch.LongTensor(features_ids))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return (loader, num_entities, words)
