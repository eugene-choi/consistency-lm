import pickle
import os
import time
import torch
import copy
import wandb
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
import numpy as np

import utils as utils
import data as data

def train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, eos_idx, args):
    stats_cache = defaultdict(list)
    
    start = time.time()
    best_val_loss = 1e5
    early_stop = 0
    '''
    B: batch size
    T: seq. length
    E: word emb. dim.
    |V|: num. of tokens in vocab set
    '''
    for epoch_number in range(args.max_epochs):
        sum_loss = 0
        sum_nll_loss = 0
        sum_os_loss = 0
        sum_os_rate = 0
        sum_non_pad_tokens = 0 #
        sum_non_pad_and_non_eos_tokens = 0 # for computing os rate.
        
        normalized_avg_eos_rank_before_T = []
        normalized_avg_eos_rank_at_T = []
        
        avg_eos_lprob_before_T = []
        avg_eos_lprob_at_T = []

        # -- training
        model.train()
        interval_start = time.time()
        for i, (inp, target) in enumerate(data_loaders["train"]):
            '''            
            inp.size(): (B,T,E)
            target.size(): (B,T)
            '''
            optimizer.zero_grad()
            inp = inp.to(device)
            target = target.to(device)
            logits = model(inp)
            # for multi-eos setting using maxout (only use it with regular lm/strlm not supported)
            if args.num_eos > 1:
                eos_maxout = torch.max(logits[:,:,eos_idx:],dim=2,keepdim=True)[0]
                logits = torch.cat([logits[:,:,:eos_idx], eos_maxout], dim=2)
            if args.loss_type == "osl":
                loss, nll_loss, os_loss, os_rate = criterion(logits, target)
            else:
                '''
                args.loss_type == "nll" or args.loss_type == "focal"
                strlm: nn.NLLLoss (requires predictions in log-prob.)
                lm: nn.CrossEntropyLoss (requires predictions in logits)
                - input args:
                1. predictions/logits.view(-1, logits.size(-1)).size(): (N,C) = (B*T,|V|)
                2. target.view(-1): (N) = (B*T)
                - returns:
                loss.size(): (N) = (B*T)
                '''
                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            # optionally we mask loss computed over the context C.
            if args.mask_context_k > 0:
                context_mask = torch.ones(target.size(), device=target.device)
                context_mask[:,:args.mask_context_k] = 0.0
                loss = (loss.view(target.size()) * context_mask).view(-1)
                if args.loss_type == "osl":
                    nll_loss = (nll_loss.view(target.size()) * context_mask).view(-1)
                    os_loss = (os_loss.view(target.size()) * context_mask).view(-1)
                    os_rate = (os_rate.view(target.size()) * context_mask).view(-1)
            
            num_context_tokens = target.size(0) * args.mask_context_k
            
            cur_non_pad_tokens = (
                target.view(-1).ne(vocab.get_id("<pad>")).nonzero().numel()
                - num_context_tokens
            ) # tensor.ne(y) : computes x \ne y element-wise.
            sum_non_pad_tokens += cur_non_pad_tokens

            loss = loss.sum()
            sum_loss += loss.item()
            loss = loss / cur_non_pad_tokens
            if args.loss_type == "osl":
                cur_non_pad_and_non_eos_tokens = cur_non_pad_tokens - target.size(0)
                sum_non_pad_and_non_eos_tokens += cur_non_pad_and_non_eos_tokens
                
                nll_loss = nll_loss.sum()
                sum_nll_loss += nll_loss.item()
                nll_loss = nll_loss / cur_non_pad_tokens
                
                os_loss = os_loss.sum()
                sum_os_loss += os_loss.item()
                os_loss = os_loss / cur_non_pad_tokens
                
                os_rate = os_rate.sum()
                sum_os_rate += os_rate.item()
                os_rate = os_rate / cur_non_pad_and_non_eos_tokens # osr only considers non-pad, non-context, non-eos tokens.
                
                # compute <eos> ranks and <eos> log-probs.
                non_pad_mask = target.ne(vocab.get_id("<pad>")).float()
                ranks_before_T, rank_T = criterion.compute_eos_ranks(logits, non_pad_mask)
                probs_before_T, probs_T = criterion.compute_eos_log_probabilities(logits, non_pad_mask)
                
                # mask out the ranks for context.
                eos_ranks_before_T = []
                eos_probs_before_T = []
                for ranks,probs in zip(ranks_before_T,probs_before_T):
                    assert len(ranks) == len(probs)
                    if len(ranks) < args.mask_context_k+1:
                        continue
                    else:
                        eos_ranks_before_T.extend(ranks[args.mask_context_k+1:])
                        eos_probs_before_T.extend(probs[args.mask_context_k+1:])
                        
                if len(eos_ranks_before_T) > 0:
                    # ranks
                    avg_eos_rank_before_T = sum(eos_ranks_before_T)/len(eos_ranks_before_T)
                    normalized_avg_eos_rank_before_T.append(avg_eos_rank_before_T / (eos_idx+1))
                    # lprobs
                    avg_eos_lprob_before_T.append(sum(eos_probs_before_T)/len(eos_probs_before_T))

                # ranks
                avg_eos_rank_at_T = sum(rank_T)/len(rank_T)
                normalized_avg_eos_rank_at_T.append(avg_eos_rank_at_T / (eos_idx+1))
                # lprobs
                avg_eos_lprob_at_T.append(sum(probs_T)/len(probs_T))
                
                log_dict={"train_inner/loss":loss,"train_inner/nll_loss":nll_loss,
                          "train_inner/os_loss":os_loss,"train_inner/os_rate":os_rate,
                          "train_inner/eos/avg_eos_rank_before_T":normalized_avg_eos_rank_before_T[-1],
                          "train_inner/eos/avg_eos_rank_at_T":normalized_avg_eos_rank_at_T[-1],
                          "train_inner/eos/avg_eos_lprob_before_T":avg_eos_lprob_before_T[-1],
                          "train_inner/eos/avg_eos_lprob_at_T":avg_eos_lprob_at_T[-1],
                         }
            else:
                log_dict={"train_inner/loss":loss,}
            
            loss.backward()
            
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            # gradient norm tracking.
            log_dict[f"train_inner/grad/grad_eos"]=0.0
            for name, param in model.named_parameters():
                log_dict[f"train_inner/grad/grad_{name}"]=param.grad.norm()
                # tracking <eos> embedding vector grad norm.
                if name == "lookup.weight":
                    log_dict[f"train_inner/grad/grad_eos"]+=param.grad[eos_idx,:].norm()
                if name == "projection.bias":
                    log_dict[f"train_inner/grad/grad_eos"]+=param.grad[eos_idx].norm()
            
            optimizer.step()
            wandb.log(log_dict)
            
            # now loss is being tracked incrementally
            if i % args.log_every == 0:
                avg_time = (time.time() - interval_start) / args.log_every
                interval_start = time.time()
                avg_loss = sum_loss / sum_non_pad_tokens
                if args.loss_type == "osl":
                    avg_nll_loss = sum_nll_loss / sum_non_pad_tokens
                    avg_os_loss = sum_os_loss / sum_non_pad_tokens
                    avg_os_rate = sum_os_rate / sum_non_pad_and_non_eos_tokens
                    
                    avg_eos_rank_before_T_log = sum(normalized_avg_eos_rank_before_T)/len(normalized_avg_eos_rank_before_T)
                    avg_eos_rank_at_T_log = sum(normalized_avg_eos_rank_at_T)/len(normalized_avg_eos_rank_at_T)
                    
                    avg_eos_lprob_before_T_log = sum(avg_eos_lprob_before_T)/len(avg_eos_lprob_before_T)
                    avg_eos_lprob_at_T_log = sum(avg_eos_lprob_at_T)/len(avg_eos_lprob_at_T)
                    
                    print("Step: %4d  ||  Train loss: %7.4f  ||  NLL loss: %7.4f  ||  OS loss: %8.4f  ||  OS rate: %7.4f  ||  EOS rank (<T): %5.4f  ||  EOS rank (T): %5.4f  ||  EOS lprob (<T): %8.4f  ||  EOS lprob (T): %8.4f  (%.2f s / step)" % (i, avg_loss, avg_nll_loss, avg_os_loss, avg_os_rate,avg_eos_rank_before_T_log,avg_eos_rank_at_T_log,avg_eos_lprob_before_T_log,avg_eos_lprob_at_T_log,avg_time))
                    
                    log_dict={"train/epoch":epoch_number,
                              "train/avg_loss":avg_loss,"train/avg_nll_loss": avg_nll_loss,
                              "train/avg_os_loss":avg_os_loss,"train/avg_os_rate":avg_os_rate,
                              "train/eos/avg_eos_rank_before_T":avg_eos_rank_before_T_log,
                              "train/eos/avg_eos_rank_at_T":avg_eos_rank_at_T_log,
                              "train/eos/avg_eos_lprob_before_T":avg_eos_lprob_before_T_log,
                              "train/eos/avg_eos_lprob_at_T":avg_eos_lprob_at_T_log,
                             }
                else:
                    print("Step: %4d  || Train loss: %7.4f\t (%.2f s / step)" % (i, avg_loss, avg_time))
                    log_dict={"train/epoch":epoch_number,"train/avg_loss":avg_loss}
                utils.log_tensorboard({"train/loss": avg_loss}, args.log_step)
                wandb.log(log_dict)
            args.log_step += 1
        
        # -- validation
        sum_valid_loss = 0
        sum_valid_nll_loss = 0
        sum_valid_os_loss = 0
        sum_valid_os_rate = 0
        sum_non_pad_tokens = 0
        sum_non_pad_and_non_eos_tokens = 0
        
        normalized_avg_eos_rank_before_T = []
        normalized_avg_eos_rank_at_T = []
        
        avg_eos_lprob_before_T = []
        avg_eos_lprob_at_T = []
        
        model.eval()
        with torch.no_grad():
            for i, (inp, target) in enumerate(data_loaders["valid"]):
                inp = inp.to(device)
                target = target.to(device)
                logits = model(inp)
                
                # for multi-eos setting using maxout.
                if args.num_eos > 1:
                    eos_maxout = torch.max(logits[:,:,eos_idx:],dim=2,keepdim=True)[0]
                    logits = torch.cat([logits[:,:,:eos_idx], eos_maxout], dim=2)
                
                if args.loss_type == "osl":
                    loss, nll_loss, os_loss, os_rate = criterion(logits, target)
                else:
                    # args.loss_type == "nll" or "focal"
                    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                    
                # optionally we mask loss computed over the context C here
                if args.mask_context_k > 0:
                    context_mask = torch.ones(target.size(), device=target.device)
                    context_mask[:,:args.mask_context_k] = 0.0
                    loss = (loss.view(target.size()) * context_mask).view(-1)
                    if args.loss_type == "osl":
                        nll_loss = (nll_loss.view(target.size()) * context_mask).view(-1)
                        os_loss = (os_loss.view(target.size()) * context_mask).view(-1)
                        os_rate = (os_rate.view(target.size()) * context_mask).view(-1)
                        
                # calculate # of <pad>
                num_context_tokens = target.size(0) * args.mask_context_k
                cur_non_pad_tokens = (
                    target.view(-1).ne(vocab.get_id("<pad>")).nonzero().numel()
                    - num_context_tokens
                )
                sum_non_pad_tokens += cur_non_pad_tokens
                
                # loss update
                loss = loss.sum()
                sum_valid_loss += loss
                if args.loss_type == "osl":
                    cur_non_pad_and_non_eos_tokens = cur_non_pad_tokens - target.size(0)
                    sum_non_pad_and_non_eos_tokens += cur_non_pad_and_non_eos_tokens
                    
                    nll_loss = nll_loss.sum()
                    sum_valid_nll_loss += nll_loss
                    
                    os_loss = os_loss.sum()
                    sum_valid_os_loss += os_loss
                    
                    os_rate = os_rate.sum()
                    sum_valid_os_rate += os_rate.sum()
                    
                    # compute <eos> ranks and <eos> log-probs before T and at T.
                    non_pad_mask = target.ne(vocab.get_id("<pad>")).float()
                    
                    ranks_before_T, rank_T = criterion.compute_eos_ranks(logits, non_pad_mask)
                    probs_before_T, probs_T = criterion.compute_eos_log_probabilities(logits, non_pad_mask)
                    
                    # mask out the ranks for context.
                    eos_ranks_before_T = []
                    eos_probs_before_T = []
                    for ranks,probs in zip(ranks_before_T,probs_before_T):
                        assert len(ranks) == len(probs)
                        if len(ranks) < args.mask_context_k+1:
                            continue
                        else:
                            eos_ranks_before_T.extend(ranks[args.mask_context_k+1:])
                            eos_probs_before_T.extend(probs[args.mask_context_k+1:])
                            
                    if len(eos_ranks_before_T) > 0:
                        # ranks
                        avg_eos_rank_before_T = sum(eos_ranks_before_T)/len(eos_ranks_before_T)
                        normalized_avg_eos_rank_before_T.append(avg_eos_rank_before_T / (eos_idx+1))
                        # lprobs
                        avg_eos_lprob_before_T.append(sum(eos_probs_before_T)/len(eos_probs_before_T))
                    
                    # ranks
                    avg_eos_rank_at_T = sum(rank_T)/len(rank_T)
                    normalized_avg_eos_rank_at_T.append(avg_eos_rank_at_T / (eos_idx+1))
                    # lprobs
                    avg_eos_lprob_at_T.append(sum(probs_T)/len(probs_T))
                        
            avg_val_loss = (sum_valid_loss / sum_non_pad_tokens).item()
            if args.loss_type == "osl":
                avg_val_nll_loss = (sum_valid_nll_loss / sum_non_pad_tokens).item()
                avg_val_os_loss = (sum_valid_os_loss / sum_non_pad_tokens).item()
                avg_val_os_rate = (sum_valid_os_rate / sum_non_pad_and_non_eos_tokens).item()
                
                normalized_avg_eos_rank_before_T_log = sum(normalized_avg_eos_rank_before_T)/len(normalized_avg_eos_rank_before_T)
                normalized_avg_eos_rank_at_T_log = sum(normalized_avg_eos_rank_at_T)/len(normalized_avg_eos_rank_at_T)
                avg_eos_lprob_before_T_log = sum(avg_eos_lprob_before_T)/len(avg_eos_lprob_before_T)
                avg_eos_lprob_at_T_log = sum(avg_eos_lprob_at_T)/len(avg_eos_lprob_at_T)
            decoding_stats = utils.decoding_dataset_stats(
                model,
                {'valid': data_loaders['valid']},
                vocab,
                device,
                num_samples={'valid': args.num_samples},
                max_steps=args.max_sample_steps,
                temperature=args.temperature,
                prefix_length=args.mask_context_k,
                decoding=("greedy",),
                consistent_sampling=False,
                one_eos=args.num_eos == 1
            )

            for data_mode_str, decode_dict in decoding_stats.items():
                for decode_mode_str, metric_dict in decode_dict.items():
                    for metric_name, val in metric_dict.items():
                        utils.log_tensorboard({f"{data_mode_str}/{decode_mode_str}/{metric_name}": val}, args.log_step)
            if args.loss_type == "osl":
                print(
                    "Epoch %3d complete.  ||  Val loss: %7.4f  ||  NLL loss: %7.4f  ||  OS loss: %7.4f  ||  OS rate: %5.4f  ||  EOS rank (<T): %5.4f  ||  EOS rank (T): %5.4f  ||  EOS lprob (<T): %6.4f  ||  EOS lprob (T): %6.4f  ||  PPL %7.4f (best %.2f)"
                    % (
                        epoch_number,
                        avg_val_loss,
                        avg_val_nll_loss,
                        avg_val_os_loss,
                        avg_val_os_rate,
                        normalized_avg_eos_rank_before_T_log,
                        normalized_avg_eos_rank_at_T_log,
                        avg_eos_lprob_before_T_log,
                        avg_eos_lprob_at_T_log,
                        utils.perplexity(avg_val_nll_loss),
                        utils.perplexity(best_val_loss),
                    )
                )
                log_dict={
                          "val/avg_val_loss":avg_val_loss,"val/avg_val_nll_loss": avg_val_nll_loss,
                          "val/avg_val_os_loss":avg_val_os_loss,"val/avg_val_os_rate":avg_val_os_rate,
                          "val/ppl":utils.perplexity(avg_val_nll_loss),
                          "val/eos/avg_eos_rank_before_T":normalized_avg_eos_rank_before_T_log,
                          "val/eos/avg_eos_rank_at_T":normalized_avg_eos_rank_at_T_log,
                          "val/eos/avg_eos_lprob_before_T":avg_eos_lprob_before_T_log,
                          "val/eos/avg_eos_lprob_at_T":avg_eos_lprob_at_T_log,
                }
            else:
                print(
                    "Epoch %d complete.\t Val loss %.4f\t PPL %.2f (best %.2f)"
                    % (
                        epoch_number,
                        avg_val_loss,
                        utils.perplexity(avg_val_loss),
                        utils.perplexity(best_val_loss),
                    )
                )
                log_dict={
                    # "val/epoch":epoch_number,
                          "val/avg_val_loss":avg_val_loss,
                          "val/ppl":utils.perplexity(avg_val_nll_loss),}
            # decoding...
            for name in decoding_stats:
                decoding_stats_ = decoding_stats[name]
                print(
                    "%s: Pct. non-term greedy %.4E (avg. len %.1f, uniq. %.4f)"
                    % (
                        name,
                        decoding_stats_["greedy"]["nonterminated"],
                        decoding_stats_["greedy"]["avg_len"],
                        decoding_stats_["greedy"]["uniq_nonterminated"],
                    )
                )
                utils.log_tensorboard(
                    {
                        '%s/greedy_%s' % (name, key): decoding_stats_['greedy'][key]
                        for key in ['nonterminated', 'avg_len']
                    },
                    args.log_step
                )
            now = time.time()
            print(
                "Total time %.1fs (%.1f)s/epoch\n"
                % ((now - start), (now - start) / (epoch_number + 1))
            )
        if args.loss_type == "osl":
            stats_cache["avg_val_loss"].append(avg_val_nll_loss)
        else:
            stats_cache["avg_loss"].append(avg_val_loss)
            
        stats_cache["decoding"].append(decoding_stats)
        
        if args.loss_type == "osl":
             avg_val_loss = avg_val_nll_loss
            
        if avg_val_loss < best_val_loss:
            utils.save(
                args, stats_cache, model, vocab, args.save_dir, "model", best=True
            )
            early_stop = 0
            best_val_loss = avg_val_loss
        else:
            early_stop += 1
            scheduler.step()
        
        log_dict["val/best_ppl"] = utils.perplexity(best_val_loss)
        log_dict["val/early_stop"] = early_stop
        
        utils.save(args, stats_cache, model, vocab, args.save_dir, "model", best=False)
        utils.log_tensorboard(
            {
                "valid/loss": avg_val_loss,
                "valid/ppl": utils.perplexity(avg_val_nll_loss),
                "valid/best_ppl": utils.perplexity(best_val_loss),
                "early_stop": early_stop,
            },
            args.log_step,
        )
        
        wandb.log(log_dict)
        
        for key in list(decoding_stats.keys()):
            decoding_stats[f"val/decoding/"] = decoding_stats.pop(key)
        
        wandb.log(decoding_stats)
        if early_stop >= args.early_stop:
            break

    print("Performing final evaluation...")
    final_eval(args, model, data_loaders, vocab, device)

def final_eval(args, model, data_loaders, vocab, device):
    ckpt = torch.load(os.path.join(args.save_dir, "model_best.pt"))
    model.load_state_dict(ckpt["model_dict"])
    model.eval()

    del data_loaders['valid']

    decoding_stats = utils.decoding_dataset_stats(
        model,
        data_loaders,
        vocab,
        device,
        num_samples={'train': 2000},
        max_steps=args.max_sample_steps,
        temperature=args.temperature,
        prefix_length=args.mask_context_k,
        decoding=("greedy", "sample",),
        one_eos=args.num_eos == 1
    )
    with open(os.path.join(args.save_dir, "final_eval.pkl"), "wb") as f:
        pickle.dump(decoding_stats, f)
    
    for key in list(decoding_stats.keys()):
        decoding_stats[f"final_eval/{key}"] = decoding_stats.pop(key)
        
    wandb.log(decoding_stats)
    pprint(decoding_stats)

def main(args):
    
    if args.dataset_version == "wikitext_sentencized":
        raw_datasets, datasets, vocab, stats = data.wikitext_sentencized(
            args.dataset_path, args.mask_context_k, args.num_eos,
        )
        args.dataset_stats = stats
    else:
        raise NotImplementedError(args.dataset_version)
        
    # Add random contexts for evaluation (given a random prefix, does STRLM terminate?)
    datasets["random"] = utils.random_prefixes(
        vocab, args.mask_context_k, args.num_samples,
    )
    
    data_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: utils.pad_collate_fn(vocab.get_id("<pad>"), x),
        )
        for name in datasets
    }
    
    eos_id = vocab.get_id("<eos>") if args.num_eos == 1 else vocab.get_id("<eos_0>")
    
    if args.model_load_dir is not None:
        print(f"Loading the model from {args.model_load_dir}...")
        
        model_args = pickle.load(open(os.path.join(args.model_load_dir, "model_best_args.pkl"), "rb"))
        vocab = pickle.load(open(os.path.join(args.model_load_dir, "model_vocab.pkl"), "rb"))
        ckpt = torch.load(os.path.join(args.model_load_dir, "model_best.pt"),map_location=torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu"))

        nested_args = getattr(model_args, 'model_args', False)
        if nested_args:
            options = nested_args.__dict__
        else:
            options = model_args.__dict__
            nested_args = model_args
            
        options['loss_type'] = args.loss_type
        options['oversmoothing_weight'] = args.oversmoothing_weight
        options['oversmoothing_margin'] = args.oversmoothing_margin
        options['label_smoothing_eps'] = args.label_smoothing_eps
        options['num_eos'] = args.num_eos
        
        model, criterion, optimizer = utils.setup_rnn(vocab, **options)
        model.load_state_dict(ckpt["model_dict"])
        args.model_args = nested_args
        args.self_terminate = args.model_args.self_terminate
        pretrain_st_epsilon = model.epsilon
        if args.overwrite_strlm == 0:
            args.st_epsilon = args.model_args.st_epsilon
        else:
            model.epsilon = args.st_epsilon
    else:
        print(f'''Initializing a scratch {"self-terminating" if args.self_terminate else "regular"} recurrent language model...''')
        model, criterion, optimizer = utils.setup_rnn(vocab, **args.__dict__)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_anneal)
    device = (torch.device("cuda:%s" % args.gpu_id) if args.gpu_id >= 0 else torch.device("cpu"))
    eos_idx = vocab.get_id("<eos>") if args.num_eos == 1 else vocab.get_id("<eos_0>")
    
    if isinstance(model, utils.RNNLanguageModelST):
        if args.model_load_dir is not None:
            model_name = "st_pt_"+str(pretrain_st_epsilon)+"_ft_"+str(model.epsilon)
        else:
            model_name = "st_"+str(model.epsilon)
    else:
        model_name = "lm"
    name = f'''{model_name}_{args.loss_type}{"_"+str(args.oversmoothing_weight) if args.loss_type == "osl" else None}_seed_{args.seed}'''
    
    print(f'\nExperiment name: {name}')
    print(f'Num <eos> tokens = {args.num_eos}')
    print(f'Model: {f"STRLM w/ ε = {model.epsilon}" if args.self_terminate == 1 else "LM"}')
    print(f'Loss: {args.loss_type} {f"w/ α = {args.oversmoothing_weight}" if args.loss_type == "osl" else ""}')
    print(f'Seed: {args.seed}\n')
    print(f'{model}\n')
    
    # initialize weights and biases
    wandb.init(project="self-terminating-rnn",name=name,mode="online" if args.wandb == 1 else "disabled")
    wandb.config.update(args)
    
    # Regular cross-entropy training:
    if args.loss_type == "nll":
        print(f'NLL loss {"training" if args.model_load_dir is None else "fine-tuning"}')
        train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, eos_idx, args)
    # Training using the oversmoothing loss:
    elif args.loss_type == "osl":
        print(f'Oversmoothing loss {"training" if args.model_load_dir is None else "fine-tuning"}')
        train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, eos_idx, args)
    elif args.loss_type == "focal":
        print(f'Focal loss {"training" if args.model_load_dir is None else "fine-tuning"}')
        train(model, vocab, data_loaders, optimizer, scheduler, criterion, device, eos_idx, args)
    else:
        raise NotImplementedError(args.loss_type)


if __name__ == "__main__":
    import argparse, os
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-base-dir", type=str, default="output")
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--expr-name", type=str, default="wikitextv2")
    parser.add_argument(
        "--dataset-path", type=str,
        default=os.path.join(Path(__file__).resolve().parents[2], "training_data/wikitext2-sentencized.json"),
    )
    parser.add_argument(
        "--include-date", action="store_true", help="include date in expr dir"
    )
    parser.add_argument(
        "--dataset-version", choices=["wikitext_sentencized"], default="wikitext_sentencized"
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rnn-dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="nn.LSTM",
        choices=["nn.RNN", "nn.LSTM", "nn.GRU"],
    )
    parser.add_argument("--tie-weights", type=int, default=1, choices=[0, 1])
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--momentum-sgd", type=float, default=0.99)
    parser.add_argument("--lr-anneal", type=float, default=0.5)
    parser.add_argument("--num-eos", type=int, default=1,
                       help="For a maxout multi-eos experiment. Only support regular LM (STRLM not supported).")

    parser.add_argument("--early-stop", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # -- loss function
    parser.add_argument(
        "--loss-type", type=str, choices=["nll", "osl","focal"], default="osl"
    )

    # -- context masking
    parser.add_argument(
        "--mask-context-k", type=int, default=10,
        help="If > 0, loss of prefix up to k is masked",
    )

    # -- self_terminating params:
    parser.add_argument("--self-terminate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--st-epsilon", type=float, default=0.0)
    
    # -- oversmoothing loss params:
    parser.add_argument("--oversmoothing-weight", type=float, default=0.5,
                       help="α term (α ∈ [0,1]) that controls the strength of the oversmoothing loss.")
    parser.add_argument("--oversmoothing-margin", type=float, default=1e-4)
    
    # -- (TODO: implement label smoothing):
    parser.add_argument("--label-smoothing-eps", type=float, default=0.0)
    
    # -- experimental params:
    parser.add_argument("--length-dist", type=int, default=0, choices=[0, 1],
                       help="For multiple <eos> runs: assigns <eos> uniformly across length dist. of seq.")

    # -- validation phase
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-sample-steps", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # -- for finetuning pretrained models
    parser.add_argument("--model-load-path", type=str, default=None)
    parser.add_argument(
        "--overwrite-strlm", type=int, default=0, choices=[0, 1],
        help="Resets the pretrained model's hyperparameter current args if set to 1."
    )
    
    # -- for enabling weights and biases
    parser.add_argument("--wandb", type=int, default=1, choices=[0, 1], help="Set 1 to use wandb.")
    
    args = utils.setup_expr(parser.parse_args())
    
    # weight tying
    if args.tie_weights == 1:
        args.embedding_dim = args.hidden_size
    
    main(args)
