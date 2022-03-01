import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import tensorboard_logger as logger
import random, copy, math
from osl import OversmoothingLoss
from focal_loss import FocalLoss

# --- data utils
class Dictionary(object):
    def __init__(
        self,
        datasets,
        include_valid=False,
        num_eos=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<cpad>", "<eos>"],
    ):
        self.tokens = []
        self.ids = {}
        self.counts = {}
        self.num_eos = num_eos
        
        for line in tqdm(datasets["train"]):
            for w in line:
                self.add_token(w)
        
        if include_valid is True:
            for line in tqdm(datasets["valid"]):
                for w in line:
                    self.add_token(w)
        
        # Add special tokens (at the end so we can optionally exclude them in the output space)
        # Always put eos last, since it is assumed by the RNNLanguageModelST
        eos_token = special_tokens[-1]
        assert len(special_tokens) > 0 and special_tokens[-1] == "<eos>"
        
        for token in special_tokens:
            if token == eos_token and num_eos > 1:
                for i in range(num_eos):
                    self.add_token(f"<eos_{i}>") # for multiple eos tokens experiments (this assigns different token ids for each eos tokens)
            else:
                self.add_token(token)

    def add_token(self, w):
        if w not in self.tokens:
            self.tokens.append(w)
            self.counts[w] = 0
            _w_id = len(self.tokens) - 1
            self.ids[w] = _w_id
        else:
            self.counts[w] += 1 # counts: dict

    def get_id(self, w):
        return self.ids[w] # ids: dict : word -> token_id

    def get_token(self, idx):
        return self.tokens[idx] # tokens: list : token_id -> word

    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]

    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids["<unk>"] for i in l] # converts the tokens in a seq. to its ids if token exists in vocab.
    
    def __len__(self):
        return len(self.tokens)


class LMDataset(Dataset):
    def __init__(self, list_of_token_lists):
        self.input_tensors = []
        self.target_tensors = []
        
        '''
        Generating LM data/label. Given a seq. with T-many tokens (<bos>, w_1, ..., w_T, <eos>) simply take:
        - input: <bos>, t_1, ..., t_T
        - label: t_1, ..., t_T-1, <eos>
        '''
        for sample in list_of_token_lists:
            self.input_tensors.append(torch.tensor([sample[:-1]], dtype=torch.long))
            self.target_tensors.append(torch.tensor([sample[1:]], dtype=torch.long))

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.target_tensors[idx]


def tokenize_dataset(datasets, dictionary, min_context_length=0, num_eos=1):
    tokenized_datasets = {}
    datasets_len = {}
    eos_stats = {}
    train_eos_index_lookup = None
    for split, dataset in datasets.items():
        print(split) # train/val/test
        _current_dictified = []
        datasets_len[split] = []
        for l in tqdm(dataset):
            datasets_len[split].append(len(l))
        eos_stats[split] = np.percentile(datasets_len[split],np.linspace(0,100,num_eos+1)[1:]).astype(int)
        
        if split == 'train':
            train_eos_index_lookup = np.zeros(eos_stats[split][-1]).astype(int)
            idx = 0
            for i in range(eos_stats[split][-1]):
                if i == eos_stats[split][idx]:
                    idx = idx + 1
                train_eos_index_lookup[i] = idx
        
        cnt_false = 0
        cnt_true = 0
        eos_0 = 0
        for l in tqdm(dataset):
            length = len(l)
            
            if len(l) < min_context_length:
                num_pad = min_context_length - len(l)
                l = ["<cpad>"] * num_pad + l
            # if num_eos == 1:
            if split == 'train':
                l = ["<bos>"] + l + [f"<eos>"]
                if len(l) == length + 2:
                    cnt_true = cnt_true +1
                else:
                    cnt_false = cnt_false + 1
            else:
                l = ["<bos>"] + l + ["<eos>"]
#             else:
#                 if split == 'train':
#                     l = ["<bos>"] + l + [f"<eos_{train_eos_index_lookup[length-1]}>"]
#                     if len(l) == length + 2:
#                         cnt_true = cnt_true +1
#                     else:
#                         cnt_false = cnt_false + 1
#                 else:
#                     l = ["<bos>"] + l + ["<eos_0>"]
            
            if l[-1] == "<eos_0>":
                eos_0 = eos_0 + 1
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
        # if split == 'train':
            # print(f'cnt_true = {cnt_true}')
            # print(f'cnt_false = {cnt_false}')
            # print(f'{len(_current_dictified) == cnt_true+cnt_false}')
            # print(f'{cnt_false/len(_current_dictified)}')
    return tokenized_datasets, eos_stats


def pad_list_of_tensors(list_of_tensors, pad_token):
    # NO truncation.
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padded_tensor = torch.cat(
            [
                t,
                torch.tensor(
                    [[pad_token] * (max_length - t.size(-1))], dtype=torch.long
                ),
            ],
            dim=-1,
        )
        padded_list.append(padded_tensor)

    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor


def pad_collate_fn(pad_idx, batch):
    input_list = [s[0] for s in batch]
    target_list = [s[1] for s in batch]
    input_tensor = pad_list_of_tensors(input_list, pad_idx)
    target_tensor = pad_list_of_tensors(target_list, pad_idx)
    return input_tensor, target_tensor


def random_prefixes(vocab, k, num_prefixes):
    if num_prefixes == -1:
        num_prefixes = 1000
    V = len(vocab)
    prefixes = torch.randint(V, (num_prefixes, k))
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<cpad>"]
    for tok in special_tokens:
        if tok in vocab.counts:
            # single_trailing_underscore_: 파이썬 키워드와의 충돌을 피하기 위해 사용하는 컨벤션이다. 그리 많이 사용하지는 않을 것이다.
            id_ = vocab.get_id(tok)
            prefixes[prefixes == id_] = 6

    bos = torch.tensor([vocab.get_id("<bos>")] * num_prefixes).unsqueeze(1)
    prefixes = torch.cat((bos, prefixes), 1).tolist()
    dataset = LMDataset(prefixes)
    return dataset


# --- model utils
class RNNLanguageModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.eos_idx = options["eos_idx"]
        self.drop = nn.Dropout(options["dropout"])
        self.lookup = nn.Embedding(
            num_embeddings=options["num_embeddings"],
            embedding_dim=options["embedding_dim"],
            padding_idx=options["padding_idx"],
        )
        '''
        The eval() function takes three parameters:
        
        eval(expression, globals=None, locals=None)

        expression - the string parsed and evaluated as a Python expression
        globals (optional) - a dictionary
        locals (optional)- a mapping object. Dictionary is the standard and commonly used mapping type in Python.
        The use of globals and locals will be discussed later in this article.
        '''
        self.rnn = eval(options["rnn_type"])(
            options["input_size"],
            options["hidden_size"],
            options["num_layers"],
            dropout=options["dropout"],
            batch_first=True,
        )
        self.projection = nn.Linear(options["hidden_size"], options["output_dim"])
        if options["tie_weights"]:
            self.projection.weight = self.lookup.weight

    def forward(self, encoded_input_sequence):
        logits, _ = self.step(encoded_input_sequence, None)
        return logits

    def step(self, encoded_input_sequence, hidden):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        output, hidden = self.rnn(embeddings, hidden)
        output = self.drop(output)
        logits = self.projection(output)
        return logits, hidden

    
    
class RNNLanguageModelST(RNNLanguageModel):
    def __init__(self, options):
        super().__init__(options)
        self.eos_idx = options["eos_idx"]
        self.epsilon = options['st_epsilon']
        self.context_length = options['mask_context_k']
    
    # for training
    def forward(self, encoded_input_sequence, return_extra=False):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        output, hidden = self.rnn(embeddings)
        output = self.drop(output)
        logits = self.projection(output) # (B,T,V)
        
        # mask is a boolean tensor with all False except for 'eos_idx' set to True.
        # mask.size() = (V)
        mask = (
            torch.arange(logits.size(2), device=logits.device, dtype=torch.long)
            == self.eos_idx
        )
        # logits = logits / math.sqrt(embeddings.size(-1))
        
        # v_logits.size() = (B,T,V-1)
        # eos_logits.size() = (B,T,1)
        v_logits = logits[:, :, ~mask]    # select non-eos logits
        eos_logits = logits[:, :, mask]   # select eos logits
        
        # p_{<eos>}^t = 1 - prod_{t' = 0}^t α(h_t')
        # σ(h_t) = (1-ε) sigmoid(u_{<eos>}.T h_t)
        # eos_logits = u_{<eos>}.T h_t , for all t = {1,..,T}
        bound = 1 - self.epsilon # (1-ε)
        # betas.size() = (B,T,1)
        betas = torch.clamp(bound*(torch.sigmoid(eos_logits)), min=1e-10)
        # betas = torch.clamp(bound*(torch.sigmoid(torch.abs(eos_logits))), min=1e-10)
        # betas = [σ(h_1), σ(h_2), σ(h_3),..., σ(h_{Τ-1}), σ(h_Τ)]
        
        # Enforce p(eos) ~= 0 for the context steps.
        if self.context_length > 0:
            betas[:, :self.context_length] = 1
        # betas = [1,1,1, ..., σ(h_{t > context_length}), ..., σ(h_{Τ-1}), σ(h_Τ)]
        # betas.size() after context mask : (B,T,1)
        
        # COMPUTING p_<eos>
        # Thm 4.1: p = 1 - prod beta
        #       => log(1-p) = sum log beta
        #       => p = 1 - exp sum log beta
        
        '''
        mine
        alphas = torch.exp(torch.cumsum(torch.log(betas), 1))
        p_eoss = 1.0 - alphas
        '''
        '''
        ilia
        p_eoss = 1.0 - torch.exp(torch.cumsum(torch.log(betas), 1))
        '''
        alphas = torch.exp(torch.cumsum(torch.log(betas), 1))
        p_eoss = 1.0 - alphas
        # torch.exp(torch.cumsum(torch.log(betas), 1)) = 
        # [1,1,1, ..., σ(h_{t > context_length}), ..., σ(h_{t > context_length})*..*σ(h_{Τ-1}), 
        # σ(h_{t > context_length})*..*σ(h_{Τ-1})*σ(h_Τ)]
        
        # p_eoss = 1.0 - torch.exp(torch.cumsum(torch.log(betas), 1)) = 
        # [0,0,0, ..., p_eos_{t > context_length}, ..., p_eos_{T-1}, p_eos_{T}]
        # p_eoss.size() : (B,T,1)
        
        
        # COMPUTING α:
        # enforce zero probability at the very first time step (to p_eos_prevs).
        # p(y_t = <eos> | y_{<t}, C) = 1 - α(h_t)
        '''
        # p_eos_prev.size() : (B,1,1)
        p_eos_prev = torch.zeros(p_eoss.size(0), 1, 1, device=p_eoss.device)
        '''
        
        '''
        p_eos_prevs = torch.cat((p_eos_prev, p_eoss), 1)[:, :-1, :]
        '''
        # p_eos_prevs = [0,0,0,0, ..., p_eos_{t > context_length}, ..., p_eos_{T-1}]
        # p_eos_prevs.size() : (B,T,1)
        
        # betas =           [1,1,1,...,   σ(h_{t > cl}),  σ(h_{t+1 > cl}), ...,      σ(h_{Τ-1}),      σ(h_Τ)]
        # (1-p_eos_prevs) = [1,1,1,1,...,             1, 1-p_eos_{t > cl}, ..., 1 - p_eos_{T-2}, 1 - p_eos_{T-1}]
        '''
        alphas = torch.clamp(betas * (1-p_eos_prevs), min=1e-10)
        '''
        ## b.c. we clip the predictive distribution with 1e-10 before outputing (nn.CELoss only accepts logits for prediction, not softmax dist. =>
        ## therefore, we can only use nn.NLLLoss which accepts prediction in the form of log-softmax b.c. our output is a piecewise dist. / not logit).
        ## If we used softmax in a regular way (passing in logits), we would not encounter a zero prob. issue with a "consistent LM" (b.c. the 
        ## activations are uniformly bounded, which prevents the predictive distribution -> delta/degenerate dist. and smooths the prob. s.t. none of
        ## them are 0 probs. -> 0 probs. in the softmax output will make log-softmax include ∞ values)
        ## B.c. we are ouputing a distribution with some probs. that has a high chance of converging into 0 or 1, we need to make sure that the
        ## output distribution contains small values such as 1e-10. o.w. nn.NLLLoss will 'nan' values (due to ∞ caused by log)
        ## Hence, below, before we output the predictive distribution, we smooth the distribution (after multiplying the softmax output by alpha)
        ## with 'min=1e-10' and then renormalize the smoothed distribution which ensures that the distribution has non-zero probabilities.
        # alphas.size() : (B,T,1)
        
        
        '''
        Question:
        Why couldn't we just set α = torch.exp(torch.cumsum(torch.log(betas), 1)) ??
        (or 
        alphas = torch.clamp(torch.exp(torch.cumsum(torch.log(betas), 1)), min=1e-10)
        with smoothing)
        
        Why go through the additional computation?
        alphas_temp = torch.clamp(torch.exp(torch.cumsum(torch.log(betas), 1)), min=1e-10)
        print(torch.norm(alphas - alphas_temp))
        print(torch.norm(alphas - alphas_temp))
        print(F.cosine_similarity(alphas, alphas_temp, dim=1, eps=1e-8))
        '''
        # alphas_temp = torch.clamp(torch.exp(torch.cumsum(torch.log(betas), 1)), min=1e-10)
        # print(torch.norm(alphas - alphas_temp))
        # print(F.cosine_similarity(alphas, alphas_temp, dim=1, eps=1e-8))
        
        
        p_Vs = alphas * torch.softmax(v_logits, -1)
        # p_Vs.size() = torch.Size([B, T, V - 1])
        
        # concatenates non-eos probs. with processed eos probs.
        ps = torch.cat((p_Vs, p_eoss), dim=2)
        # ps.size() = torch.Size([B, T, V])
        # print(f'eos before smoothing = {p_eoss}')
        # print(f'eos before smoothing = {torch.norm(p_eoss)}')
        
        
        # clamps the predicted distribution such that the min. prob is 1e-10
        # smoothing the predictive distribution
        ps = torch.clamp(ps, min=1e-10)
        # ps.size() = torch.Size([B, T, V])
        
        # re-normalize the distribution after clamping/smoothing
        ps = ps / ps.sum(-1, keepdim=True)
        
        # output the distribution in a log-scale. (b.c. nn.NLLLoss only accepts log-distribution)
        log_ps = torch.log(ps)
        # log_ps.size() = torch.Size([B, T, V])
        if return_extra:
            return log_ps, hidden, p_eoss[:, -1:, :]
        return log_ps
    
    # for evaluation
    def step(self, encoded_input_sequence, hidden, p_eos_prev=None):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        output, hidden = self.rnn(embeddings, hidden)
        output = self.drop(output)
        logits = self.projection(output)
        mask = (
            torch.arange(logits.size(2), device=logits.device, dtype=torch.long)
            == self.eos_idx
        )

        v_logits = logits[:, :, ~mask]
        eos_logits = logits[:, :, mask]

        bound = 1 - self.epsilon
        betas = torch.clamp(bound*(torch.sigmoid(eos_logits)), min=1e-10)
        # betas = torch.clamp(bound*(torch.sigmoid(torch.abs(eos_logits))), min=1e-10)
        # -- Difference for the step case
        '''
        we use step for evaluation. for training, teacher forcing is possible so we could have 
        '''
        alphas = torch.clamp(betas * (1-p_eos_prev), min=1e-10)
        p_eos = 1 - alphas
        p_Vs = alphas * torch.softmax(v_logits, -1)

        ps = torch.cat((p_Vs, p_eos), dim=2)
        ps = torch.clamp(ps, min=1e-10)
        ps = ps / ps.sum(-1, keepdim=True)
        log_ps = torch.log(ps)
        return log_ps, hidden, p_eos

class STRLMPretrainFinetune(RNNLanguageModelST):
    def __init__(self, options):
        super().__init__(options)
        self.pretrain = options['pretrain']
        if options['break_emb_weights']:
            self.projection.weight = copy.deepcopy(self.lookup.weight)

    # for training
    def forward(self, encoded_input_sequence, return_extra=False):
        if self.pretrain:
            return F.log_softmax(RNNLanguageModel.forward(self, encoded_input_sequence), dim=-1)
        else:
            return RNNLanguageModelST.forward(self, encoded_input_sequence, return_extra)
        
    def step(self, encoded_input_sequence, hidden, p_eos_prev=None):
        if self.pretrain:
            logits, hidden = RNNLanguageModel.step(self, encoded_input_sequence, hidden)
            return F.log_softmax(logits, dim=-1), hidden
        else:
            return RNNLanguageModelST.step(self, encoded_input_sequence, hidden, p_eos_prev)

        
        
class RNNLanguageModelSTEOS(RNNLanguageModel):
    def __init__(self, options):
        super().__init__(options)
        self.eos_idx = options["eos_idx"]
        self.epsilon = options['st_epsilon']
        self.context_length = options['mask_context_k']
        self.eos = nn.GRU(input_size=options['embedding_dim'],
                          hidden_size=options['hidden_size'],
                          num_layers=1,
                          batch_first=True,
                         )
        self.eos_param = nn.Linear(options['hidden_size'],options['embedding_dim'])
    # for training
    def forward(self, encoded_input_sequence, return_extra=False):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        output, hidden = self.rnn(embeddings)
        eos, _ = self.eos(output)
        eos_params = self.eos_param(eos)
        output = self.drop(output)
        eos_logits = torch.sum(eos_params * output, dim=2, keepdim=True)
        logits = self.projection(output) # (B,T,V)
        
        # mask is a boolean tensor with all False except for 'eos_idx' set to True.
        # mask.size() = (V)
        mask = (
            torch.arange(logits.size(2), device=logits.device, dtype=torch.long)
            == self.eos_idx
        )
        # v_logits.size() = (B,T,V-1)
        # eos_logits.size() = (B,T,1)
        v_logits = logits[:, :, ~mask]    # select non-eos logits
        return torch.cat((v_logits, eos_logits), dim=2)
    
    # for evaluation
    def step(self, encoded_input_sequence, hidden, eos_hidden, p_eos_prev=None):
        embeddings = self.drop(self.lookup(encoded_input_sequence))
        output, hidden = self.rnn(embeddings, hidden)
        eos, eos_hidden = self.eos(output, eos_hidden)
        eos_params = self.eos_param(eos)
        output = self.drop(output)
        eos_logits = torch.sum(eos_params * output, dim=2, keepdim=True)
        logits = self.projection(output)
        mask = (
            torch.arange(logits.size(2), device=logits.device, dtype=torch.long)
            == self.eos_idx
        )

        v_logits = logits[:, :, ~mask]
        return torch.cat((v_logits, eos_logits), dim=2), hidden, eos_hidden
        
    
def setup_rnn(vocab, **options):
    options_ = {
        "num_embeddings": len(vocab),
        "output_dim": len(vocab),
        "embedding_dim": 64,
        "padding_idx": vocab.get_id("<pad>"),
        "eos_idx": vocab.get_id("<eos>" if options['num_eos'] == 1 else "<eos_0>"),
        "hidden_size": 128,
        "num_layers": 2,
        "rnn_dropout": 0.1,
        "lr": 0.001,
        "rnn_type": "nn.RNN",
    }
    for k, v in options.items():
        options_[k] = v
    if options_["tie_weights"]:
        options_["input_size"] = options_["embedding_dim"]
    if options_["self_terminate"]:
        return setup(RNNLanguageModelST, options_)
    else:
        return setup(RNNLanguageModel, options_)


# --- eval utils
def perplexity(avg_loss):
    ppl = 2 ** (avg_loss / np.log(2))
    return ppl


def decoding_dataset_stats(
    model,
    dataloaders,
    vocab,
    device,
    num_samples={},
    max_steps=500,
    temperature=1.0,
    prefix_length=5,
    decoding=("greedy", "sample"),
    progress_bar=False,
    consistent_sampling=False,
    save_decoding_logs=False,
    one_eos=False,
):
    results = {}

    def _stats(sequences):
        # uniq_nonterminated is the ratio of nonterminated decodings that are unique. if all the nonterminated decodings are different
        # from one another, the ratio 1.0. If there are non-terminated decodings that are decoded in the same way, then the ratio is
        # less than 1.0
        non_terminated = [tuple(x) for x in sequences if len(x) == max_steps]
        s = {
            "nonterminated": len(non_terminated) / len(sequences),
            "uniq_nonterminated": len(set(non_terminated))
            / max(len(non_terminated), 1),
            "avg_len": np.mean([len(x) for x in sequences]),
        }
        return s

    def _to_text(prefixes, res, targets, vocab):
        pr_texts = []
        cont_texts = []
        target_texts = []
        for pr, cont, target in zip(prefixes, res, targets):
            pr_text = vocab.decode_idx_seq(pr)
            cont_text = vocab.decode_idx_seq(cont)
            target_text = vocab.decode_idx_seq(target)
            pr_texts.append(pr_text)
            cont_texts.append(cont_text)
            target_texts.append(target_text)
        
        return (pr_texts, cont_texts, target_texts) 
    
    if isinstance(model, STRLMPretrainFinetune):
        print(f'''model.pretrain = {model.pretrain}''')
        model.pretrain = False
        print(f'''model.pretrain = {model.pretrain}''')

    print("Computing decoding stats...")
    for name, data_loader in dataloaders.items():
        results[name] = {}
        for mode in decoding:
            print("dataset %s\tdecoding %s" % (name, mode))
            res, prefixes, targets = decode_dataset(
                model,
                data_loader,
                vocab.get_id("<bos>"),
                vocab.get_id("<eos>" if one_eos else "<eos_0>"),
                num_samples.get(name, -1),
                max_steps,
                mode=mode,
                device=device,
                prefix_length=prefix_length,
                temperature=temperature,
                progress_bar=progress_bar,
                consistent_sampling=consistent_sampling,
            )
            
            key = mode if isinstance(mode, str) else "%s-%s" % (mode[0], str(mode[1]))
            results[name][key] = _stats(res)
            if save_decoding_logs:
                results[name][key]['decoding_logs'] = _to_text(prefixes, res, targets, vocab)
    return results


def compute_ppl_dataloader(
    args,
    vocab,
    model,
    criterion,
    data_loader,
    device,
    progress_bar=False
):

    iterator = (
            tqdm(enumerate(data_loader), total=len(data_loader))
            if progress_bar
            else enumerate(data_loader)
        )
    model.eval()
    sum_valid_loss = 0
    non_pad_tokens = 0

    with torch.no_grad():
        for i, (inp, target) in iterator:
            inp = inp.to(device)
            target = target.to(device)
            logits = model(inp)
            '''
            logits.view(-1, logits.size(-1)).size() : (batch_size * seq. length, num_vocab)
            target.size(batch_size, seq. length, 1)
            target.view(-1, logits.size(-1)).size() : (batch_size * seq. length)
            '''
            if args.loss_type == "osl":
                loss, nll_loss, os_loss, logging_output = criterion(logits, target)
                loss = nll_loss
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            # optionally we mask loss computed over the context C here
            if args.mask_context_k > 0:
                context_mask = torch.ones(target.size(), device=target.device)
                context_mask[:, : args.mask_context_k] = 0.0
                loss = loss.view(target.size()) * context_mask
                loss = loss.view(-1)
            
            loss = loss.sum()
            sum_valid_loss += loss

            num_context_tokens = target.size(0) * args.mask_context_k
            current_non_pad_tokens = (
                target.view(-1).ne(vocab.get_id("<pad>")).nonzero().numel()
                - num_context_tokens
            )
            non_pad_tokens += current_non_pad_tokens
        avg_loss = sum_valid_loss.item() / non_pad_tokens
    return avg_loss

def decode_dataset(
    model,
    data_loader,
    bos_token,
    eos_token,
    num_samples,
    max_steps,
    mode,
    device,
    prefix_length=0,
    temperature=1.0,
    progress_bar=False,
    consistent_sampling=False,
):
    with torch.no_grad():
        xs = []
        prefixes = []
        targets = []
        iterator = (
            tqdm(enumerate(data_loader), total=len(data_loader) if num_samples == -1 else num_samples//data_loader.batch_size)
            if progress_bar
            else enumerate(data_loader)
        )
        for minibatch_id, (inp, target) in iterator:
            inp = inp.to(device)
            # encode the prefix
            hidden = None
            p_eos_prev = None
            eos_hidden = None
            prefix = inp[:, : prefix_length + 1]  # +1 for <bos>
            if isinstance(model, STRLMPretrainFinetune):
                if model.pretrain:
                    output, hidden = model.step(prefix, hidden)
                else:
                    output, hidden, p_eos_prev = model(prefix, return_extra=True)
            elif isinstance(model, RNNLanguageModelST):
                output, hidden, p_eos_prev = model(prefix, return_extra=True)
            elif isinstance(model, RNNLanguageModelSTEOS):
                output, hidden, eos_hidden = model.step(prefix,hidden,eos_hidden)
            else:
                output, hidden = model.step(prefix, hidden)

            if 'beam' in mode:
                beam_size = int(mode.split("_")[1])  # e.g. beam_4
                batch_size = inp.size(0)
                max_timestep = max_steps

                count_finished = torch.zeros(batch_size, device=device).long()
                finished_hypotheses = {
                    i:[] for i in range(batch_size)
                }
                finished_scores = {
                    i:[] for i in range(batch_size)
                }

                # first beam iteration is out of the loop here
                log_probs = torch.log_softmax(output[:, -1, :], dim=-1)  # (batch_size, vocab_size)
                vocab_size = log_probs.size(-1)

                top_scores, top_tokens = torch.topk(log_probs, beam_size, dim=-1, largest=True, sorted=True)

                # we add to finished even now when eos is selected too
                current_eos_mask = (top_tokens == eos_token)
                count_finished = count_finished + current_eos_mask.sum(1).long()

                # we need this loop... ?
                for beam_id, beam_eos_mask in enumerate(current_eos_mask):
                    if any(beam_eos_mask):
                        finished_in_this_beam = top_tokens[beam_id, beam_eos_mask]
                        finished_scores_in_this_beam = top_scores[beam_id, beam_eos_mask]
                        finished_hypotheses[beam_id].extend([finished_in_this_beam.tolist()])
                        finished_scores[beam_id].extend(finished_scores_in_this_beam.tolist())
                
                hypotheses = [
                    (
                        top_tokens[:,:,None],
                        top_scores,
                        torch.zeros_like(top_tokens)
                    )
                ]

                # expanding the hidden tuple up to the beam_size
                if isinstance(hidden, tuple):  # LSTM
                    expanded_hidden = [None,None]
                    for i in range(2):
                        expanded_hidden[i] = hidden[i][:,:,None,:].expand(-1,-1,beam_size,-1).reshape(2,batch_size*beam_size,-1)
                    expanded_hidden = tuple(expanded_hidden)
                else:
                    expanded_hidden = hidden[:,:,None,:].expand(-1,-1,beam_size,-1).reshape(2,batch_size*beam_size,-1)

                # input for the first beam timestep
                expanded_input = top_tokens.view(batch_size*beam_size,1)
                for timestep in range(1, max_timestep):
                    # change below should be enough for the STRNN
                    expanded_output, expanded_hidden = model.step(expanded_input, expanded_hidden)

                    # reshaping back as batch_size * beam_size
                    decoupled_output = expanded_output[:, None, :,:].view(batch_size, beam_size, 1, -1)  # (batch, beam, 1, vocab)
                    # -> log_softmax
                    decoupled_output = torch.log_softmax(decoupled_output, dim=-1)

                    partial_from_prev_timestep = hypotheses[timestep-1][0]  # index 0 is partial
                    scores_from_prev_timestep = hypotheses[timestep-1][1]  # index 1 is scores
                    
                    # check for eos, do not select anything after eos
                    eos_mask = partial_from_prev_timestep[:,:,-1] == eos_token
                    scores_from_prev_timestep[eos_mask] = -10e15

                    extended_scores = decoupled_output.add(scores_from_prev_timestep[:,:,None,None])

                    # coupling it beam*vocab for topk
                    coupled_extended_scores = extended_scores.view(batch_size, beam_size*vocab_size)
                    top_scores, top_ids = torch.topk(coupled_extended_scores, beam_size, dim=-1, largest=True, sorted=True)
                    
                    actual_word_ids = top_ids % vocab_size
                    
                    # make a new input for next iteration

                    expanded_input = actual_word_ids.view(batch_size*beam_size, -1)

                    prev_hyp_id_per_sample = top_ids // vocab_size

                    prev_hyp_id_flat = ((torch.arange(batch_size, device=device) * beam_size)[:,None] + prev_hyp_id_per_sample).view(-1)
                    reordered_prev_hypotheses = torch.index_select(partial_from_prev_timestep.view(batch_size*beam_size,-1), dim=0, index=prev_hyp_id_flat).view(batch_size, beam_size, -1)
                    extended_current_hypotheses = torch.cat([reordered_prev_hypotheses, actual_word_ids[:,:,None]], dim=2)

                    # check currently extended hyps for eos
                    current_eos_mask = (actual_word_ids == eos_token)
                    count_finished = count_finished + current_eos_mask.sum(1).long()

                    # we need this loop... ?
                    for beam_id, beam_eos_mask in enumerate(current_eos_mask):
                        if any(beam_eos_mask):
                            finished_in_this_beam = extended_current_hypotheses[beam_id, beam_eos_mask, :]
                            finished_scores_in_this_beam = top_scores[beam_id, beam_eos_mask]
                            finished_hypotheses[beam_id].extend(finished_in_this_beam.tolist())
                            finished_scores[beam_id].extend(finished_scores_in_this_beam.tolist())

                    # reorder the hidden state
                    if isinstance(expanded_hidden, tuple):  # LSTM
                        new_expanded_hidden = [None,None]
                        num_layers = expanded_hidden[0].size(0)
                        for i in range(num_layers):
                            
                            new_expanded_hidden[i] = torch.index_select(expanded_hidden[i], dim=1, index=prev_hyp_id_flat)
                        new_expanded_hidden = tuple(expanded_hidden)
                    else:
                        new_expanded_hidden = torch.index_select(expanded_hidden, dim=1, index=prev_hyp_id_flat)
                    expanded_hidden = new_expanded_hidden

                    # add new hypotheses to beam
                    hypotheses.append(
                        (extended_current_hypotheses, top_scores, prev_hyp_id_per_sample)
                    )
                    # check if we have enough ( at least 1) finished for each sample in mini batch
                    # ideally one would do at least beam size, with 1 avg len might be shorter
                    if all(count_finished > 0):
                        break
                
                # now we check what hypotheses are finished
                best_finished_seqs = []
                for beam_id in range(batch_size):
                    if count_finished[beam_id].item() == 0:
                        # non-terminated here
                        # take the first seq from the beam
                        seq = hypotheses[-1][0][beam_id][0].cpu().tolist()
                    else:
                        # find the best one w.r.t score
                        finished_here = finished_scores[beam_id]
                        best_finished_id = np.array(finished_here).argmax()
                        seq = finished_hypotheses[beam_id][best_finished_id]
                    best_finished_seqs.append(seq) 
                x = best_finished_seqs

            else:
                # decode
                x = []
                p_eoss = []
                output = output[:, -1, :].unsqueeze(1)
                for t in range(max_steps):

                    if mode == "greedy":
                        xt = output.argmax(-1) # greedy is deterministic approx. decoding algo. therefore we use argmax.
                    elif mode == "sample":
                        if isinstance(model, RNNLanguageModelST) and temperature != 1.0:
                            raise NotImplementedError
                        elif isinstance(model, RNNLanguageModelST):
                            xt = output.exp().squeeze(1).multinomial(1)
                        else:
                            xt = (
                                torch.softmax(output / temperature, -1)
                                .squeeze(1)
                                .multinomial(1)
                            )
                    elif isinstance(mode, tuple):
                        if mode[0] == "topk":
                            output = top_k_top_p_filtering(output.squeeze(1), top_k=mode[1], 
                                consistent_sampling=consistent_sampling, eos_idx=model.eos_idx)
                        elif mode[0] == "topp":
                            output = top_k_top_p_filtering(output.squeeze(1), top_p=mode[1], 
                                consistent_sampling=consistent_sampling, eos_idx=model.eos_idx)
                        xt = torch.softmax(output, -1).multinomial(1) 
                        # topp is a sampling-based algorithm (non-deterministic approx. decoding algo.) 
                        # therefore we use multinomial(1) to sample the most likely token from the predicted distribution.
                    if isinstance(model, STRLMPretrainFinetune):
                        if model.pretrain:
                            output, hidden = model.step(xt, hidden)
                        else:
                            output, hidden, p_eos_prev = model.step(xt, hidden, p_eos_prev)
                            p_eoss.append(p_eos_prev)
                    elif isinstance(model, RNNLanguageModelST):
                        output, hidden, p_eos_prev = model.step(xt, hidden, p_eos_prev)
                        p_eoss.append(p_eos_prev)
                    elif isinstance(model, RNNLanguageModelSTEOS):
                        output, hidden, eos_hidden = model.step(prefix,hidden,eos_hidden)
                    else:
                        output, hidden = model.step(xt, hidden)

                    x.append(xt)
                x = torch.cat(x, 1)
                
            prefixes.append(prefix)
            
            if isinstance(x, torch.Tensor):
                xs.append(x)
            else:
                xs.extend(x)
                
            if isinstance(target, torch.Tensor):
                target = F.pad(target, (0, max_steps - target.size(1)), "constant", 0)
                targets.append(target)
            else:
                targets.append(target)
            if num_samples >= 0 and (minibatch_id + 1) * inp.size(0) > num_samples:
                break
        if isinstance(xs[0], torch.Tensor):
            xs = torch.cat(xs, 0).tolist()
        
        prefixes = torch.cat(prefixes, 0).tolist()
        if isinstance(targets[0], torch.Tensor):
            targets = torch.cat(targets, 0).tolist()
        # targets = torch.cat(targets, 0).tolist()
        
        for i, x in enumerate(xs):
            if eos_token in x:
                xs[i] = x[: x.index(eos_token)]
        for i, x in enumerate(targets):
            if eos_token in x:
                targets[i] = x[: x.index(eos_token)]
        return xs, prefixes, targets


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1, 
    consistent_sampling=False, eos_idx=None
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        if consistent_sampling:
            indices_to_remove[:, eos_idx].fill_(False)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        if consistent_sampling:
            indices_to_remove[:, eos_idx].fill_(False)

        logits[indices_to_remove] = filter_value
    return logits


# --- experiment utils
def setup(model_class, options):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        current_device = "cuda"
    else:
        current_device = "cpu"
    '''
    dictionary.get() method:
    - provides a default value if the key is missing:
        dictionary.get(key, default_value)
    returns default_value (whatever you choose it to be), whereas
        dictionary["bogus"]
    would raise a KeyError.
    - If omitted, default_value is None, such that
        dictionary.get("bogus")  # <-- No default specified -- defaults to None
    returns None just like dictionary.get("bogus", None) would.
    '''
    if options.get("model_load_path", None) is not None:
        model_load_path = options["model_load_path"]
        if not os.path.exists(model_load_path):
            raise EOFError("Invalid model load path %s" % model_load_path)
        if torch.cuda.is_available():
            model_dict = torch.load(model_load_path)
        else:
            model_dict = torch.load(model_load_path, map_location=torch.device('cpu'))
        model = model_class(options).to(current_device)
        model.load_state_dict(model_dict["model_dict"])
    else:
        model = model_class(options).to(current_device)
    
    if options["loss_type"] == "osl":
        criterion = OversmoothingLoss(strlm=options['self_terminate'] == 1,
                                      pad_idx=options["padding_idx"],
                                      eos_idx=options["eos_idx"],
                                      label_smoothing_eps=options["label_smoothing_eps"],
                                      oversmoothing_weight=options["oversmoothing_weight"],
                                      oversmoothing_margin=options["oversmoothing_margin"],)
    elif options["loss_type"] == "nll":
        if options["self_terminate"]:
            criterion = nn.NLLLoss(ignore_index=options["padding_idx"], reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=options["padding_idx"], reduction="none")
    elif options["loss_type"] == "focal":
        criterion = FocalLoss(ignore_index=options["padding_idx"],
                              gamma=6,
                              reduction='none')
    else:
        raise NotImplementedError("invalid loss function")
    
    model_parameters = [p for p in model.parameters() if p.requires_grad]

    if options["optimizer"] == "adam":
        optimizer = optim.Adam(model_parameters, lr=options["lr"])
    elif options["optimizer"] == "sgd":
        optimizer = optim.SGD(model_parameters, lr=options["lr"], momentum=options['momentum_sgd'])
    else:
        raise NotImplementedError("invalid optimizer")

    return model, criterion, optimizer


def save(args, stats_cache, model, vocab, save_dir, name, best, log=True):
    name = "%s%s" % (name, "_best" if best else "")
    torch.save({"model_dict": model.state_dict()}, os.path.join(save_dir, name + ".pt"))
    pickle.dump(stats_cache, open(os.path.join(save_dir, name + "_stats.pkl"), "wb"))
    pickle.dump(args, open(os.path.join(save_dir, name + "_args.pkl"), "wb"))
    pickle.dump(vocab, open(os.path.join(save_dir, name + "_vocab.pkl"), "wb"))
    if log:
        print("Model saved: %s" % name)


def setup_tensorboard(args):
    log_directory = args.save_dir
    args.log_step = 1
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    try:
        logger.configure(log_directory)
    except ValueError:
        pass


def log_tensorboard(values_dict, step):
    for k, v in values_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            logger.log_value(k, v, step)


def _expr_dir(args, include_date=False):
    from datetime import datetime

    now = datetime.now()
    d = os.path.join(
        args.save_base_dir,
        args.expr_name,
        ""
        if not include_date
        else "%d_%d_%d_%d_%d" % (now.year, now.month, now.day, now.hour, now.minute),
    )
    i = 2
    d_ = d
    while os.path.exists(d_):
        d_ = d + "_" + str(i)
        i += 1
    return d_


def setup_expr(args):
    args.save_dir = _expr_dir(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_tensorboard(args)
    print("Save dir: %s" % args.save_dir)
    return args
