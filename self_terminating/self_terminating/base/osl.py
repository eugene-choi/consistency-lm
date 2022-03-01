from dataclasses import dataclass, field
import torch
import torch.nn as nn
import logging, math

logger = logging.getLogger(__name__)

def compute_oversmoothing_logratio(lprobs, target, non_pad_mask, eos_idx, margin=1e-4):
    '''
    lprobs: (B,T,|V|)
    target: (B,T)
    non_pad_mask: (B,T)
    '''
    '''
    Gathers values along an axis specified by dim.
    target_lprobs:(B,T,1)
    '''
    target_lprobs = torch.gather(lprobs, dim=-1, index=target.unsqueeze(-1))
    
    # reverse cumsum fast workaround, this makes approximation error for suffix_lprob[:,-1]
    # in other words, after this operation the smallest suffix of one token does not equal exactly to that
    # true eos_probability. So it is better to exlcude those positions from OSL since theoretically loss there is 0.
    '''
    * target_lprobs_withoutpad: (B,T)
    - computes the joint conditional distribution of the tokens at the current timestep conditioned upon all the previous tokens.
        (for all seq. in B) [log p(y_1), log p(y_2|y_1), log p(y_3|y_2,y_1), ... , log p(y_T|y_<T)]
        
    * torch.sum(target_lprobs_withoutpad, dim=-1, keepdims=True): (B,1)
    - computes the log probability of the current full sequence.
        (for B = 1) [log p(y_1,y_2,y_3,...,y_T-1,y_T)]
        
    * torch.cumsum(target_lprobs_withoutpad, dim=-1): (B,T)
    - computes the log joint distribution of the tokens up until the current timestep.
        (for B = 1) [log p(y_1), log p(y_1,y_2), log p(y_1,y_2,y_3), ... , log p(y_1,y_2,y_3,...,y_T)]
    
    * suffix_lprob: (B,T)
    - the log joint conditional probability of continuing the ground truth sequence given the previous tokens.
    - THIS IS THE QUANTITY THAT WE WANT TO MAXIMIZE/PUSH UP ON UNTIL y_T for every sequence.
    
    * eos_lprobs: (B,T)
    - selects the log conditional probability of the <eos> token at each timestep, conditioned upon all the previous tokens.
    - THIS IS THE QUANTITY THAT WE WANT TO MINIMIZE/PUSH DOWN ON UNTIL y_T for every sequence.
    
    * oversmoothing_loss (1): (B,T)
    - oversmoothing losses for each sequence at each timestep.
    
    * oversmoothing_loss (2): (scalar)
    - oversmoothing loss of the current training batch.
    '''
    # First, we exclude the predicted log probabilities of the paddings.
    target_lprobs_withoutpad = (target_lprobs.squeeze(-1) * non_pad_mask)
    # Then, compute $\sum_{t'=1}^{|\mathbf{y}|} \log p(y_{t'} \mid y_{<t'})$ ('target_lprobs_withoutpad' term is the approximation term; it is not needed from the mathematical derivation)
    suffix_lprob = target_lprobs_withoutpad + torch.sum(target_lprobs_withoutpad, dim=-1, keepdims=True) - torch.cumsum(target_lprobs_withoutpad, dim=-1)
    # selects the log conditional probability of the <eos> token at each timestep for every sequence (batch index).
    eos_lprobs = lprobs[:,:,eos_idx] * non_pad_mask
    oversmoothing_loss = torch.maximum(eos_lprobs - suffix_lprob + margin, torch.zeros_like(suffix_lprob))
    # oversmoothing_loss.sum(dim=1)  : (B) -> o.s.l. for each sequence in the current training batch.
    # non_pad_mask.squeeze(dim=-1).sum(dim=1) : (B) -> sequence lengths (w.o. pads) for each sequences in the current training batch.
    
    ## oversmoothing_loss = (oversmoothing_loss.sum(dim=1) / non_pad_mask.squeeze(dim=-1).sum(dim=1)).mean() ##
    
    '''
    # computing the oversmoothing "rate" here for free
    oversmoothed: (B,T)
    '''
    with torch.no_grad():
        # counts/marks the timesteps where the log conditional probabilities of the <eos> tokens 
        # are greater than that of the suffix log probabilities
        oversmoothed = eos_lprobs > suffix_lprob
        # exclude pad cases from oversmoothing rate
        oversmoothed = oversmoothed * non_pad_mask
        # exclude y_t=<eos> from oversmoothing counts
        oversmoothed = oversmoothed * (target != eos_idx).float() 
        
        
        # exclude the <eos> from each seq count
        # num_osr_per_seq: (B)
        '''
        num_osr_per_seq = non_pad_mask.sum(-1) - 1
        '''
        
        # compute oversmoothing per sequence
        '''
        osr = oversmoothed.sum(-1) / num_osr_per_seq
        '''
        
        # os_rate: (B,T) tensor of 0,1.
        os_rate = oversmoothed
        # print(f'os_rate[0,:]={os_rate[0,:]}')
    return oversmoothing_loss, os_rate

'''
@: python decorator

dataclasses 모듈에서 제공하는 @dataclass 데코레이터를 일반 클래스에 선언해주면 해당 클래스는 소위 데이터 클래스가 됩니다.
데이터 클래스는 __init__(), __repr__(), __eq__()와 같은 메서드를 자동으로 생성해줍니다. 
'''
@dataclass
class OversmoothingLoss(nn.Module):
    oversmoothing_weight: float = field(
        default=0.5,
        metadata={'help': '(1-os_weight)*NLL + os_weight*OSL'}
    )
    oversmoothing_margin: float = field(
        default=0,
        metadata={'help': 'max(log p(eos|prefix) - log p(suffix|prefix) + M, 0)'}
    )
    
    def __init__(self, strlm:bool, pad_idx:int, eos_idx:int, label_smoothing_eps:float, oversmoothing_weight=0.5, oversmoothing_margin=1e-4):
        super().__init__()
        self.strlm = strlm
        self.compute_loss = nn.NLLLoss(ignore_index=pad_idx, reduction="none") if strlm else nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.label_smoothing_eps = label_smoothing_eps
        self.oversmoothing_margin = oversmoothing_margin
        self.oversmoothing_weight = oversmoothing_weight
        logger.info(f'Oversmoothing loss, margin={oversmoothing_margin}, weight={oversmoothing_weight}')
 
    def forward(self, lprobs, target):
        '''
        lprobs (log probabilities if self-terminating else logits): (B,T,|V|)
        target: (B,T)
        '''
        size = lprobs.size()
        '''
        ** nn.NLLLoss requires input size (N,|V|) and target size (N).
        ** b.c. we set 'ignore_index=pad_idx' when instantiating nn.NLLLoss, nll_loss[target.view(-1) == self.pad_idx].sum() is equal to 0.
        nll_loss: (B*T)
        '''
        nll_loss = self.compute_loss(lprobs.view(-1, size[-1]), target.view(-1))
            
        if not self.strlm:
            lprobs = lprobs.log_softmax(dim=-1)
        
        # non_pad_mask: (B,T)
        non_pad_mask = (target != self.pad_idx).float()
        nll_loss = nll_loss.view(size[0],size[1])
        '''
        nll_loss: (B,T)
        os_loss: (B,T)
        os_rate: (B) [computed per sequence]
        loss: (B,T)
        '''
        os_loss, os_rate = compute_oversmoothing_logratio(lprobs, target, non_pad_mask, self.eos_idx, self.oversmoothing_margin)
        loss = (1.0-self.oversmoothing_weight)*nll_loss + self.oversmoothing_weight*os_loss
        return loss, nll_loss, os_loss, os_rate
    
    @torch.no_grad()
    def compute_eos_log_probabilities(self, model_logit, non_pad_mask):
        model_lprobs = model_logit if self.strlm else model_logit.log_softmax(dim=-1)
        eos_model_lprobs = model_lprobs[:,:,self.eos_idx]
        real_eos_ids = non_pad_mask.sum(dim=1).long() - 1  # convert lengths to eos ids
        probs_before_last_ts = []
        probs_last_ts = []
        
        for probs_seq, real_eos_id in zip(eos_model_lprobs, real_eos_ids):
            probs_before_last_ts.append(probs_seq[:real_eos_id].tolist())
            probs_last_ts.append(probs_seq[real_eos_id].item())

        return probs_before_last_ts, probs_last_ts

    @torch.no_grad()
    def compute_eos_ranks(self, model_logit, non_pad_mask):
        # select <eos> logit/log-prob for all timestep t and compare it against the logit/log-prob values 
        # of all tokens in the vocab for each timestep t. Then sum up the boolean result to get <eos> rank.
        eos_ranks = (model_logit[:,:,self.eos_idx].unsqueeze(dim=-1) >= model_logit[:,:]).float().sum(dim=-1)
        # eos_ranks = (model_logit[:,:,self.eos_idx].unsqueeze(dim=-1) <= model_logit[:,:]).float().sum(dim=-1)
        
        real_eos_ids = non_pad_mask.sum(dim=1).long() - 1  # convert lengths to eos ids
        # <eos> rank before t=T (y_T^* = <eos>)
        ranks_before_last_ts = []
        rank_last_ts = []
        for rank_seq, real_eos_id in zip(eos_ranks, real_eos_ids):
            ranks_before_last_ts.append(rank_seq[:real_eos_id].tolist())
            rank_last_ts.append(rank_seq[real_eos_id].item())
        return ranks_before_last_ts, rank_last_ts
    
    
    @torch.no_grad()
    def compute_eos_ranks(self, model_logit, non_pad_mask):
        # select <eos> logit/log-prob for all timestep t and compare it against the logit/log-prob values 
        # of all tokens in the vocab for each timestep t. Then sum up the boolean result to get <eos> rank.
        eos_ranks = (model_logit[:,:,self.eos_idx].unsqueeze(dim=-1) >= model_logit[:,:]).float().sum(dim=-1)
        # eos_ranks = (model_logit[:,:,self.eos_idx].unsqueeze(dim=-1) <= model_logit[:,:]).float().sum(dim=-1)
        
        real_eos_ids = non_pad_mask.sum(dim=1).long() - 1  # convert lengths to eos ids
        # <eos> rank before t=T (y_T^* = <eos>)
        ranks_before_last_ts = []
        rank_last_ts = []
        for rank_seq, real_eos_id in zip(eos_ranks, real_eos_ids):
            ranks_before_last_ts.append(rank_seq[:real_eos_id].tolist())
            rank_last_ts.append(rank_seq[real_eos_id].item())
        return ranks_before_last_ts, rank_last_ts



# class LabelSmoothingLossCanonical(nn.Module):
#     def __init__(self, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLossCanonical, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             true_dist = torch.zeros_like(pred)
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#             true_dist += self.smoothing / pred.size(self.dim)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
 


    
#     @torch.no_grad()
#     def valid_step(self, sample, model, criterion):
#         # original valid step
#         model.eval()
#         with torch.no_grad():
#             loss, sample_size, logging_output = criterion(model, sample)

#         # processing target sequences
#         non_pad_mask = (sample['target'] != 1).float()[:,:,None]
#         target_net_output = model(**sample['net_input'])
        
#         target_model_lprobs = target_net_output[0].log_softmax(dim=-1)
#         target_model_true_lprobs = torch.gather(target_model_lprobs, dim=-1, index=sample['target'].unsqueeze(-1))
#         target_eos_log_probs_nt, target_eos_log_probs_t = self.compute_eos_log_probabilities(target_net_output[0], non_pad_mask.squeeze(-1))
#         target_false_eos_ranks, target_true_eos_ranks = self.compute_eos_ranks(target_net_output[0], non_pad_mask.squeeze(-1))
#         target_seq_lengths = non_pad_mask.squeeze(-1).sum(-1).tolist()
#         logging_output['target_eos_log_probs_nt'] = target_eos_log_probs_nt
#         logging_output['target_eos_log_probs_t'] = target_eos_log_probs_t
#         logging_output['target_false_eos_ranks'] = target_false_eos_ranks
#         logging_output['target_true_eos_ranks'] = target_true_eos_ranks
#         logging_output['target_seq_lengths'] = target_seq_lengths
#         logging_output['target_model_log_probs'] = target_model_true_lprobs.detach().cpu().tolist()
#         logging_output['target_src_seq_lengths'] = sample['net_input']['src_lengths'].tolist()