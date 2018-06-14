import torch
import json
from pos_emb import PositionalEmbedding
from loader import get_loader
from config import LR, WINDOW_SIZE
from nltk.corpus import gutenberg, abc, brown, state_union
import argparse
import time

corpora = {
    'abc': abc,
    'brown': brown,
    'gutenberg': gutenberg,
    'state_union': state_union,
}

timestamp = int(time.time())
parser = argparse.ArgumentParser()
parser.add_argument('--corpus')
parser.add_argument('--regularization')
parser.add_argument('--reg_weight', type=float)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--max_dim', type=int)
parser.add_argument('--num_runs', type=int)
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
MAX_DIM = args.max_dim
NUM_RUNS = args.num_runs
corpus = corpora[args.corpus]


def process_batch(emb, batch, optim):
    optim.zero_grad()
    loss = emb(batch)
    loss.backward()
    optim.step()
    return loss


def train_epoch(emb, optim, loader):
    loss = 0
    for batch in loader:
        loss += process_batch(emb, batch, optim)
    return loss


loader, num_entities, words = get_loader(corpus, timestamp)

for run_id in range(NUM_RUNS):
    f = open('final_emb_test.txt', 'a')
    prior_emb = None
    prior_model = None
    for dim in range(1, MAX_DIM + 1, 1):
        print(dim)
        step_emb = PositionalEmbedding(dim=dim, num_entities=num_entities,
                                       window_size=WINDOW_SIZE,
                                       prior_emb=prior_emb,
                                       prior_model=prior_model, cuda=True,
                                       regularization_type=args.regularization,
                                       regularization_weight=args.reg_weight)
        optim = torch.optim.Adam(step_emb.parameters(), lr=LR)
        for _ in range(NUM_EPOCHS):
            loss = train_epoch(step_emb, optim, loader)
            print(loss)
            prior_emb = step_emb.get_prior_emb()
            prior_model = step_emb.get_prior_model()

    final_emb = step_emb.get_prior_emb()

    ids = range(num_entities)
    for ent_id, ent_emb in zip(ids, final_emb.weight.detach().cpu().numpy().tolist()):
        f.write("%d\t%s\t%s\t%.6f\t%d\t%d\t%.8f\t%d\t%d\t%d\t%s\t%s\n" % (
            timestamp,
            args.corpus,
            args.regularization,
            args.reg_weight,
            args.num_epochs,
            args.max_dim,
            LR, WINDOW_SIZE,
            run_id,
            ent_id,
            words[ent_id],
            json.dumps(ent_emb))
        )
    f.close()
