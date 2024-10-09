import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Selection_model(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        if model_params['pooling']:
            self.encoder = Encoder_h(**model_params)
            feature_dim = 2 * model_params['embedding_dim'] + 1
        else:
            self.encoder = Naive_Encoder(**model_params)
            feature_dim = model_params['embedding_dim'] + 1

        if self.model_params['ns_feature'] == True:
            model_params_ = model_params.copy()
            model_params_['embedding_dim'] = feature_dim
            self.representative_net = representative_net(**model_params_)
            self.init_tokens = nn.Parameter(torch.Tensor(1, feature_dim))
            self.init_tokens.data.uniform_(-1, 1)
            self.model_tokens = []
            feature_dim = 2 * feature_dim
            self.similarity = nn.Sequential(
                nn.Linear(feature_dim, model_params['embedding_dim']),
                nn.GELU(),
                nn.Linear(model_params['embedding_dim'], 1))
            
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, model_params['embedding_dim']),
                nn.GELU(),
                nn.Linear(model_params['embedding_dim'], model_params['output_dim']))

    def acquire_feature(self, points, scales, mask):
        graph_emb = self.encoder(points, mask)
        instance_feature = torch.cat((
            graph_emb,
            scales[:, None]
        ), dim=1)
        return instance_feature
    
    def update_tokens(self, representative_features):
        self.model_tokens = []  # reset
        for i in range(self.init_tokens.shape[0]):
            self.model_tokens.append(self.representative_net(self.init_tokens[i], representative_features[i]))

    def forward(self, points, scales, manual_features, mask, return_feature=False):
        graph_emb = self.encoder(points, mask)

        # shape: (batch, problem, EMBEDDING_DIM)
        if manual_features == None:
            manual_features = scales[:, None]
        else:
            manual_features = torch.cat((manual_features, scales[:, None]), dim=-1)

        self.instance_feature = torch.cat((
            graph_emb,
            manual_features
        ), dim=1)

        if self.model_params['ns_feature'] == True:
            probs = []
            batch_size = self.instance_feature.shape[0]
            for i in range(len(self.model_tokens)):
                probs.append(self.similarity(torch.cat((self.instance_feature, self.model_tokens[i][None, :].expand(batch_size, -1)), dim=-1)))
            probs = torch.cat(probs, dim=-1)
        else:
            probs = self.classifier(self.instance_feature)

        return probs

class representative_net(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.att_layer_1 = EncoderLayer(**model_params)
        self.att_layer_2 = EncoderLayer(**model_params)
    
    def forward(self, model_token, representative_features):
        # arange dimensions
        model_tokens = model_token[None, None, :]
        representative_features = representative_features.unsqueeze(0)
        input1 = torch.cat((model_tokens, representative_features), dim=1)
        # layer 1: self-attention
        out1 = self.att_layer_1(input1)
        # layer 2: representative ins -> token
        out2 = self.att_layer_1(out1[:, :1, :], kv=out1[:, 1:, :])

        return out2.squeeze(0).squeeze(0)

class Naive_classifier(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        if self.model_params['problem_type'] == 'TSP':
            feature_dim = 10
        else:
            feature_dim = 12
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, model_params['embedding_dim']),
            nn.GELU(),
            nn.Linear(model_params['embedding_dim'], model_params['output_dim']))

    def forward(self, points, scales, features, mask):
        features = torch.cat((features, scales[:, None]), dim=-1)
        probs = self.classifier(features)
 
        return probs

class Naive_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        node_dim = 2
        
        if model_params['problem_type'] == 'TSP':
            self.embedding = nn.Linear(node_dim, embedding_dim)
        if model_params['problem_type'] == 'CVRP':
            self.embedding = nn.Linear(node_dim + 1, embedding_dim)
        
        encoder_layer_num = self.model_params['encoder_layer_num'] * self.model_params['block_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
    
    def forward(self, points, mask):
        # points shape: (batch, n', node_dim)
        # mask shape: (batch, n', n')
        embs = self.embedding(points)

        for layer in self.layers:
            embs = layer(embs, mask=mask)

        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        graph_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return graph_emb

class Encoder_h(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        node_dim = 2
        if model_params['problem_type'] == 'TSP':
            self.embedding = nn.Linear(node_dim, embedding_dim)
        if model_params['problem_type'] == 'CVRP':
            self.embedding = nn.Linear(node_dim + 1, embedding_dim)
        self.blocks = nn.ModuleList([Encoder_block_h(**model_params) for _ in range(model_params['block_num'])])
        self.nonlinear = nn.GELU()

    def masked_mean(self, embs, mask):
        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return mean_emb
    
    def masked_max(self, embs, mask):
        # Mask embs of padded nodes as -inf
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]

        return max_emb

    def forward(self, data, mask):
        # data.shape: (batch, problem, 2)
        out = self.embedding(data)
        # shape: (batch, problem, embedding)

        # Hierachy embedding
        i = 0
        graph_emb_h = 0.
        for block in self.blocks:
            graph_emb, out, mask = block(out, mask, i)
            graph_emb_h += graph_emb
            i += 1

        mean_emb = self.masked_mean(out, mask)
        max_emb = self.masked_max(out, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        # shape: (batch, 2 * embedding)
        graph_emb_h += graph_emb

        return graph_emb

class Encoder_block_h(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.layer_score = EncoderLayer(**model_params)
        self.p = nn.Linear(model_params['embedding_dim'], 1)
        self.modulate = nn.Linear(1, model_params['embedding_dim'])
        self.act = nn.Tanh()
        self.nonlinear = nn.GELU()
    
    def padding_concate(self, list_):
        lengths = torch.tensor([t.shape[0] for t in list_], device=list_[0].device)
        max_len = lengths.max().item()
        mask = torch.zeros(len(list_), max_len)
        for i, t in enumerate(list_):
            list_[i] = F.pad(t, (0, 0, 0, max_len - lengths[i]))[None, :, :]
            mask[i, lengths[i]: max_len] = float('-inf')
        embs = torch.cat(list_, dim=0)

        return embs, mask

    def masked_mean(self, embs, mask):
        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return mean_emb
    
    def masked_max(self, embs, mask):
        # Mask embs of padded nodes as -inf
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]

        return max_emb

    def forward(self, embs, mask, i):
        # embs shape: (batch, n', embedding)
        # adj_mat shape: (batch, n', n')
        # mask shape: (batch, n', n')

        for layer in self.layers:
            embs = layer(embs, mask)
        
        mean_emb = self.masked_mean(embs, mask)
        max_emb = self.masked_max(embs, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        # shape: (batch, 2 * embedding)

        # Downsampling
        score_embs = self.layer_score(embs, mask)
        scores = self.act(self.p(score_embs))
        scores = scores.squeeze(-1)
        scores = scores + mask

        selected_embs_list = []
        unselected_embs_list = []
        lengths = (mask == 0).sum(dim=-1, keepdim=True)
        for i in range(embs.shape[0]):
            score, ind = scores[i].topk(int(lengths[i] * self.model_params['downsample_ratio']), dim=-1, largest=True)
            selected_emb = embs[i].take_along_dim(ind[:, None].expand(-1, embs.shape[-1]), dim=0)
            selected_emb = selected_emb + score[:, None]
            # selected_emb = selected_emb + self.act(self.modulate(score[:, None]))
            selected_embs_list.append(selected_emb)


        # Pad and cat
        selected_embs, selected_mask = self.padding_concate(selected_embs_list)

        return graph_emb, selected_embs, selected_mask

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, mask=None, edges=None, kv=None):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        if kv is None:
            kv = input1
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(kv), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v, rank2_ninf_mask=mask)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(-1)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
    assert not score_scaled.isinf().all(dim=-1).any(), "All the valid nodes are filtered! Check the pooling operation."

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
        
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    
    return out_concat

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2
        else:
            back_trans = input1 + input2

        return back_trans

class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
