import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
from os import PathLike
import torch
import gzip, csv
from os import PathLike
from sentence_transformers.readers import InputExample


def plot_attributions(A, tokens_a, tokens_b, 
    size: Tuple = (7, 7), 
    dst_path: Optional[PathLike] = None,
    show_colorbar: bool = False,
    cmap: str = 'RdBu',
    range: Optional[float] = None,
    shrink_colorbar: float = 1.,
    bbox = None
):
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    assert isinstance(A, np.ndarray)
    Sa, Sb = A.shape
    assert len(tokens_a) == Sa and len(tokens_b) == Sb, 'size mismatch of tokens and attributions'
    if range is None:
        range = np.max(np.abs(A))
    f = plt.figure(figsize=size)
    plt.imshow(A, cmap=cmap, vmin=-range, vmax=range)
    plt.yticks(np.arange(A.shape[0]), labels=tokens_a)
    plt.xticks(np.arange(A.shape[1]), labels=tokens_b, rotation=50, ha='right')
    if show_colorbar:
        plt.colorbar(shrink=shrink_colorbar)
    if dst_path is not None:
        plt.savefig(dst_path, bbox_inches=bbox)
        plt.close()
    else:
        return f


def input_to_device(inpt: dict, device: torch.device):
    for k, v in inpt.items():
        if isinstance(v, torch.Tensor):
            inpt[k] = v.to(device)


def load_sts_data(path: PathLike):
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples
