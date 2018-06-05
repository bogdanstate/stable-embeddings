import torch
from nltk.corpus import brown as corpus
import json
from pos_emb import PositionalEmbedding
from loader import get_loader
from config import LR, NUM_EPOCHS, MAX_DIM, NUM_RUNS, WINDOW_SIZE


def process_batch(emb, batch, optim):
    optim.zero_grad()
    loss = emb(batch)
    loss.backward()
    optim.step()
    return loss


def train_epoch(emb, optim, loader):
    loss = 0
    for batch, in loader:
        loss += process_batch(emb, batch, optim)
    return loss


loader, num_entities = get_loader(corpus)

for run_id in range(NUM_RUNS):
    f = open('final_emb.txt', 'a')
    prior_emb = None
    prior_model = None
    for dim in range(1, MAX_DIM + 1, 1):
        print(dim)
        step_emb = PositionalEmbedding(dim=dim, num_entities=num_entities,
                                       window_size=WINDOW_SIZE,
                                       prior_emb=prior_emb,
                                       prior_model=prior_model, cuda=True)
        optim = torch.optim.Adam(step_emb.parameters(), lr=LR)
        for _ in range(NUM_EPOCHS):
            loss = train_epoch(step_emb, optim, loader)
            print(loss)
            prior_emb = step_emb.get_prior_emb()
            prior_model = step_emb.get_prior_model()

    final_emb = step_emb.get_prior_emb()
    ids = range(num_entities)
    for ent_id, ent_emb in zip(ids, final_emb.weight.detach().cpu().numpy().tolist()):
        f.write("%d\t%d\t%s\n" % (run_id, ent_id, json.dumps(ent_emb)))
    f.close()
