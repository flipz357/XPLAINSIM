from datasets import load_dataset
from sentence_transformers.models import Pooling
import torch
from tqdm import tqdm
import pickle
from os.path import exists, join
from os import makedirs
import random

import sys
sys.path.append('../')
from xsbert import models


def init_model(model_name: str = 'sbert', device: str = 'cuda:0'):
    if model_name == 'sbert':
        model_name = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
        transformer = models.ReferenceTransformer(model_name)
        pooling = Pooling(transformer.get_word_embedding_dimension())
        model = models.XSRoberta(modules=[transformer, pooling])
    elif model_name == 'jina':
        model_name = 'jinaai/jina-embeddings-v2-base-en'
        transformer = models.ReferenceTransformer(model_name)
        pooling = Pooling(transformer.get_word_embedding_dimension())
        model = models.XJina(modules=[transformer, pooling])
    else:
        print(f'model name {model_name} not recognized, options are: sbert and jina')
    print(f'initializing model: {model_name}')
    model.to(torch.device(device))
    return model


def compute_attributions(
    samples: list[tuple], 
    model: torch.nn.Module, 
    device: str,
    n_steps: int = 10
):
    model.reset_attribution()
    model.init_attribution_to_layer(idx=10, N_steps=n_steps)
    results = []
    for pair in tqdm(samples):
        query = pair[0]
        passage = pair[1]
        A, tokens_q, tokens_p = model.explain_similarity(
            query, 
            passage, 
            move_to_cpu=True,
            sim_measure='cos',
            device=device,
            verbose=False
        )
        results.append({
            'query': query,
            'passage': passage,
            'query_tokens': tokens_q,
            'passage_tokens': tokens_p,
            'attribution': A
        })
    return results


def sample_pairs(dataset: list, n_samples: int = 1000, positives_only: bool = False):
    pairs = []
    sample_indices = random.sample(range(0, len(dataset)), k=n_samples)
    for i in sample_indices:
        inst = dataset[i]
        q = inst['query']
        passages = inst['passages']['passage_text']
        if positives_only:
            passages = [
                p for p, l in zip(passages, inst['passages']['is_selected'])
                if l == 1
            ]
        try:
            p = random.choice(passages)
        except IndexError:
            print(f'no passages found for index: {i}, skipping')
            continue
        pairs.append((q, p))
    return pairs


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='sbert', 
        help='model name, options are: sbert and jina')
    parser.add_argument("--n-integration-steps", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--directory", type=str, default='data/')
    parser.add_argument("--attributions-file", type=str, default='attributions.pkl')
    parser.add_argument("--positives-only", type=bool, default=False)
    parser.add_argument("--msmarco-version", type=str, default="v1.1",
        help='version of the msmarco dataset to use, options are: v1.1 and v2.1')
    parser.add_argument("--dataset-split", type=str, default="test", 
        help='split of the msmarco dataset to use, options are: train, validation and test')
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()

    if not exists(args.directory):
        makedirs(args.directory)

    print('init model')
    model = init_model(args.model_name, device=args.device)

    print('load dataset')
    ds = load_dataset("microsoft/ms_marco", args.msmarco_version)

    print('sampling pairs')
    pairs = sample_pairs(
        dataset=ds[args.dataset_split],
        n_samples=args.n_samples, 
        positives_only=args.positives_only
    )

    print('computing attributions')
    results = compute_attributions(
        samples=pairs, 
        model=model, 
        device=args.device,
        n_steps=args.n_integration_steps
    )

    print('saving results')
    pickle.dump(
        results,
        open(join(args.directory, args.attributions_file), 'wb')
    )