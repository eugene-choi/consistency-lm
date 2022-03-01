import numpy as np
import utils as utils
import pickle
import json
import os


def wikitext_sentencized(path, min_context_length=0, num_eos=1):
    raw_datasets = json.load(open(path, 'r'))
    for key in raw_datasets:
        raw_datasets[key] = [x.split() for x in raw_datasets[key]]
    vocab_file = '.vocab.p' if num_eos == 1 else f'.vocab_{num_eos}.p'
    
    if os.path.exists(path + vocab_file):
        vocab = pickle.load(open(path+vocab_file, 'rb'))
    else:
        vocab = utils.Dictionary(raw_datasets, include_valid=True, num_eos=num_eos)
        pickle.dump(vocab, open(path+vocab_file, 'wb')) # Write a pickled representation of obj to the open file object file.

    tokenized_datasets, eos_stats = utils.tokenize_dataset(raw_datasets, vocab, min_context_length=min_context_length, num_eos=num_eos)
    datasets = {name: utils.LMDataset(ds) for name, ds in tokenized_datasets.items()}
    stats = {'path': path,
             'num_train': len(raw_datasets['train']),
             'num_valid': len(raw_datasets['valid']),
             'vocab_size': len(vocab),
             'avg_len': np.mean([len(d) for d in raw_datasets['train']])}
    print("Dataset loaded.\n\tTrain size: %d\tValid size: %d\n\t|V|: %d\tmax len: %d\tavg len: %d\n" % (
            len(raw_datasets['train']),
            len(raw_datasets['valid']),
            len(vocab),
            max(len(x) for x in raw_datasets['train']),
            stats['avg_len']))
    return raw_datasets, datasets, vocab, stats

