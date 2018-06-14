import torch


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, dim, num_entities, window_size,
                 prior_emb=None, cuda=True, prior_model=None,
                 regularization_type=None, regularization_weight=0):

        super(PositionalEmbedding, self).__init__()
        self.dim = dim
        self.num_entities = num_entities
        self.window_size = window_size
        self.prior_emb = prior_emb
        self.emb = torch.nn.EmbeddingBag(num_entities, 1, mode='sum')
        self.model = torch.nn.Linear(1, num_entities)
        self.prior_model = prior_model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.regularization_type = regularization_type
        self.regularization_weight = regularization_weight

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

    def get_regularization_term(self):

        if self.regularization_type == 'L1':
            return self.emb.weight.abs().sum()
        if self.regularization_type == 'L2':
            return self.emb.weight.pow(2).sum()
        if self.regularization_type == 'prev_abs':
            if self.dim == 1:
                return 0
            return (
                # current abs val > 0.1
                (self.emb.weight.abs() - 0.1).sign().clamp(min=0) *
                # prior abs val < 0.1
                (1-(self.prior_emb.weight.abs() - 0.1).sign().clamp(min=0))
            ).sum()
        return 0

    def forward(self, batch):
        focal_ids, features_ids = batch

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

        loss += self.regularization_weight * self.get_regularization_term()

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
