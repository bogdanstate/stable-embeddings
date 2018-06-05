import torch


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, dim, num_entities, window_size,
                 prior_emb=None, cuda=True, prior_model=None):

        super(PositionalEmbedding, self).__init__()
        self.dim = dim
        self.num_entities = num_entities
        self.window_size = window_size
        self.prior_emb = prior_emb
        self.emb = torch.nn.EmbeddingBag(num_entities, 1, mode='sum')
        self.model = torch.nn.Linear(1, num_entities)
        self.prior_model = prior_model
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if cuda:
            self.cuda = True
            self.emb = self.emb.cuda()
            self.model = self.model.cuda()
            if self.prior_emb is not None:
                self.prior_emb = self.prior_emb.cuda()
            if self.prior_model is not None:
                self.prior_model = self.prior_model.cuda()

        if self.prior_emb is not None:
            self.prior_emb.weight = torch.nn.Parameter(
                self.prior_emb.weight.detach(),
                requires_grad=False
            )
        if self.prior_model is not None:
            self.prior_model.weight = torch.nn.Parameter(
                self.prior_model.weight.detach(),
                requires_grad=False
            )
        self.add_module('emb', self.emb)
        self.add_module('model', self.model)

    def parameters(self):

        return ([x for x in self.emb.parameters()] +
                [x for x in self.model.parameters()])

    def forward(self, batch):
        focal_ids = batch[:, [self.window_size]]
        features_ids = batch[
            :,
            [x for x in range(0, 2 * self.window_size + 1)
             if x != self.window_size]
        ]

        if self.cuda:
            focal_ids = focal_ids.cuda()
            features_ids = features_ids.cuda()

        out = self.emb(features_ids)
        out = self.model(out)

        if self.prior_emb is not None:
            step_out = self.prior_emb(features_ids)
            step_out = self.prior_model(step_out)
            out = step_out + out
        loss = self.loss_fn(input=out, target=focal_ids.squeeze())
        return(loss)

    def get_prior_emb(self):
        emb = torch.nn.EmbeddingBag(self.num_entities, self.dim, mode='sum')
        if self.prior_emb is not None:
            emb.weight = torch.nn.Parameter(
                torch.cat((self.prior_emb.weight, self.emb.weight), dim=1),
                requires_grad=False
            )
        else:
            emb.weight = self.emb.weight
        return(emb)

    def get_prior_model(self):
        model = torch.nn.Linear(self.dim, self.num_entities)
        if self.prior_model is not None:
            model.weight = torch.nn.Parameter(
                torch.cat((self.prior_model.weight, self.model.weight), dim=1),
                requires_grad=False
            )
        else:
            model.weight = self.model.weight
        return model
