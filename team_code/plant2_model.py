"""
PlanT2 model (HFLM - HuggingFace Language Model for driving).
Adapted from https://github.com/autonomousvision/plant2
"""

import logging

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
)

logger = logging.getLogger(__name__)


class _DictAccessor:
    """Wrap a dict to allow attribute access (handles nested OmegaConf-style configs)."""
    def __init__(self, d):
        self._d = d
    def __getattr__(self, name):
        if name == '_d':
            return super().__getattribute__('_d')
        val = self._d[name]
        if isinstance(val, dict):
            return _DictAccessor(val)
        return val
    def get(self, key, default=None):
        return self._d.get(key, default)


class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        if isinstance(config_all, dict):
            config_all = _DictAccessor(config_all)
        if isinstance(config_net, dict):
            config_net = _DictAccessor(config_net)
        self.config_all = config_all
        self.config_net = config_net

        self.object_types = 7
        self.num_attributes = 6
        self.fc_attributes = 4

        precisions = [
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_angle", 4),
            self.config_all.model.pre_training.get("precision_speed", 4),
        ]

        self.vocab_size = [2**i for i in precisions]
        self.vocab_size[0] = int((1 + self.config_all.model.training.get("range_factor_front", 1))/2*self.vocab_size[0])

        config = AutoConfig.from_pretrained(self.config_net.hf_checkpoint)
        self.n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        self.model.embeddings.word_embeddings = None
        self.model.pooler = None

        self.input_bev = self.config_all.model.training.get("input_bev", False)
        if self.input_bev:
            import timm
            self.bev_encoder = timm.create_model("resnet18", pretrained=True, num_classes=512)

        self.tok_emb = nn.ParameterList(
            nn.Linear(self.num_attributes, self.n_embd)
            for _ in range(self.object_types)
        )

        self.wp_rep = self.config_all.model.waypoints.representation
        self.wp_gen = self.config_all.model.waypoints.generator

        self.wp_len = self.config_all.model.waypoints.wps_len if self.wp_rep != "path+2hot" else 0
        self.path_len = self.config_all.model.waypoints.path_len if self.wp_rep != "waypoints" else 0

        if self.wp_gen == "singlegru":
            num_tokens = 1 if self.wp_rep == "waypoints" else 2
        elif self.wp_rep == "path+2hot":
            num_tokens = self.path_len + 1
        elif self.wp_rep == "path+wps":
            num_tokens = self.wp_len + self.path_len
        elif self.wp_rep == "waypoints":
            num_tokens = self.wp_len

        self.wp_token = nn.Parameter(torch.randn(num_tokens, self.n_embd))

        if self.config_net.get("use_dropout", False):
            self.drop = nn.Dropout(config_net.embd_pdrop)

        self.route_emb = nn.Linear(20*2, self.n_embd)
        self.speed_emb = nn.Embedding(4, self.n_embd)

        self.input_ego_speed = self.config_all.model.training.get("input_ego_speed", False)
        if self.input_ego_speed:
            self.ego_speed_emb = nn.Linear(1, self.n_embd)

        self.heads = nn.ModuleList(
            [nn.Linear(self.n_embd, n_out) for n_out in self.vocab_size]
        )

        if self.wp_rep != "path+2hot":
            if self.wp_gen == "linear":
                self.wp_generator = LinearWaypoints(self.n_embd)
            elif self.wp_gen == "multigru":
                self.wp_generator = GRUWaypointsPredictorInterFuser(
                    self.n_embd, self.wp_len,
                    self.config_all.model.waypoints.gru_hidden_size, 0)
            elif self.wp_gen == "singlegru":
                self.wp_generator = SingleGRUWaypoints(self.n_embd, self.wp_len)

        if self.wp_rep != "waypoints":
            if self.wp_gen == "linear":
                self.path_generator = LinearWaypoints(self.n_embd)
            elif self.wp_gen == "multigru":
                self.path_generator = GRUWaypointsPredictorInterFuser(
                    self.n_embd, self.path_len,
                    self.config_all.model.waypoints.gru_hidden_size, 0)
            elif self.wp_gen == "singlegru":
                self.path_generator = SingleGRUWaypoints(self.n_embd, self.path_len)

        if self.wp_rep == "path+2hot":
            self.speed_classifier = nn.Linear(
                self.n_embd, self.config_all.model.waypoints.bins_speed)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        batch_idxs = batch["idxs"]
        x_batch_objs = batch["x_objs"]
        route_batch = batch["route_original"]
        speed_limit_batch = batch["speed_limit"]

        embedding = torch.zeros((*x_batch_objs.shape[:-1], self.n_embd), device=x_batch_objs.device)
        for i in range(len(self.tok_emb)):
            mask = x_batch_objs[..., 0] == i
            if mask.any():
                embedding[mask] = self.tok_emb[i](x_batch_objs[mask, 1:])

        embedding = embedding[batch_idxs]

        route_tok = self.route_emb(route_batch.flatten(1))[:, None]
        embedding = torch.cat((route_tok, embedding), dim=1)

        speed_tok = self.speed_emb(speed_limit_batch)[:, None]
        embedding = torch.cat((speed_tok, embedding), dim=1)

        remove_idxs = 2

        if self.input_ego_speed:
            ego_speed_tok = self.ego_speed_emb(batch["input_ego_speed"][:, None])[:, None]
            embedding = torch.cat((ego_speed_tok, embedding), dim=1)
            remove_idxs += 1

        if self.input_bev:
            bev_tok = self.bev_encoder(batch["BEV"])[:, None]
            embedding = torch.cat((bev_tok, embedding), dim=1)
            remove_idxs += 1

        wp_tokens = self.wp_token.expand(embedding.shape[0], *self.wp_token.shape)
        embedding = torch.cat((wp_tokens, embedding), dim=1)
        remove_idxs += self.wp_token.shape[0]

        if self.config_net.get("use_dropout", False):
            embedding = self.drop(embedding)

        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        if batch.get("y_objs") is not None:
            targets = batch["y_objs"][batch_idxs]
            targets = [targets[..., i].flatten() for i in range(self.fc_attributes)]
            logits = x[:, remove_idxs:]
            logits = [self.heads[i](logits).flatten(end_dim=-2) for i in range(self.fc_attributes)]
        else:
            targets = None
            logits = None

        pred_path = None
        pred_wps = None
        pred_speed = None

        if self.wp_gen == "singlegru":
            if self.wp_rep != "path+2hot":
                pred_wps = self.wp_generator(x[:, 0, :])
            if self.wp_rep != "waypoints":
                pred_path = self.path_generator(x[:, 1, :])
        else:
            if self.wp_rep != "path+2hot":
                pred_wps = self.wp_generator(x[:, :self.wp_len, :])
            if self.wp_rep != "waypoints":
                pred_path = self.path_generator(x[:, self.wp_len:self.wp_len+self.path_len, :])

        if self.wp_rep == "path+2hot":
            if self.wp_gen == "singlegru":
                pred_speed = self.speed_classifier(x[:, 0, :])
            else:
                pred_speed = self.speed_classifier(x[:, self.path_len, :])

        pred_plan = (pred_path, pred_wps, pred_speed)
        return logits, targets, pred_plan, attn_map


class SingleGRUWaypoints(nn.Module):
    def __init__(self, n_embd, num_wps):
        super().__init__()
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=2, hidden_size=64)
        self.wp_output = nn.Linear(64, 2)
        self.num_wps = num_wps

    def forward(self, token):
        z = self.wp_head(token)
        output_wp = []
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype, device=z.device)
        for _ in range(self.num_wps):
            z = self.wp_decoder(x, z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)
        return torch.stack(output_wp, dim=1)


class LinearWaypoints(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.wp_decoder = nn.Linear(n_embd, 2)

    def forward(self, tokens):
        diffs = self.wp_decoder(tokens)
        return torch.cumsum(diffs, 1)


class GRUWaypointsPredictorInterFuser(nn.Module):
    def __init__(self, input_dim, waypoints, hidden_size, target_point_size):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        if target_point_size > 0:
            self.encoder = nn.Linear(target_point_size, hidden_size)
        self.target_point_size = target_point_size
        self.hidden_size = hidden_size
        self.decoder = nn.Linear(hidden_size, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point=None):
        bs = x.shape[0]
        if self.target_point_size > 0:
            z = self.encoder(target_point).unsqueeze(0)
        else:
            z = torch.zeros((1, bs, self.hidden_size), device=x.device)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)
        return output