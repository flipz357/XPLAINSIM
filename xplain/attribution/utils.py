import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, Any
from os import PathLike
import torch
import gzip
import csv
from sentence_transformers.readers import InputExample


def plot_attributions(
    attributions_matrix: torch.Tensor,
    tokens_a: list,
    tokens_b: list,
    size: Tuple[int, int] = (7, 7),
    dst_path: Optional[PathLike] = None,
    show_colorbar: bool = False,
    cmap: str = "RdBu",
    range: Optional[float] = None,
    shrink_colorbar: float = 1.0,
    bbox: Optional[Any] = None,
) -> Optional[plt.Figure]:
    """Plots the attribution matrix with tokens on x and y axes.

    Args:
        attributions_matrix (torch.Tensor): The matrix containing attribution values.
        tokens_a (list): List of tokens corresponding to the rows.
        tokens_b (list): List of tokens corresponding to the columns.
        size (Tuple[int, int], optional): Figure size. Defaults to (7, 7).
        dst_path (Optional[PathLike], optional): Path to save the figure. Defaults to None.
        show_colorbar (bool, optional): Whether to display the colorbar. Defaults to False.
        cmap (str, optional): Colormap. Defaults to "RdBu".
        range (Optional[float], optional): Value range for visualization. Defaults to None.
        shrink_colorbar (float, optional): Factor to shrink the colorbar. Defaults to 1.0.
        bbox (Optional[Any], optional): Bounding box for saving the figure. Defaults to None.

    Returns:
        Optional[plt.Figure]: The plotted figure if not saving.
    """
    if isinstance(attributions_matrix, torch.Tensor):
        attributions_matrix = attributions_matrix.numpy()

    assert isinstance(attributions_matrix, np.ndarray)
    Sa, Sb = attributions_matrix.shape
    assert (
        len(tokens_a) == Sa and len(tokens_b) == Sb
    ), "Size mismatch of tokens and attributions"

    if range is None:
        range = np.max(np.abs(attributions_matrix))

    f = plt.figure(figsize=size)
    plt.imshow(attributions_matrix, cmap=cmap, vmin=-range, vmax=range)
    plt.yticks(np.arange(Sa), labels=tokens_a)
    plt.xticks(np.arange(Sb), labels=tokens_b, rotation=50, ha="right")

    if show_colorbar:
        plt.colorbar(shrink=shrink_colorbar)

    if dst_path is not None:
        plt.savefig(dst_path, bbox_inches=bbox)
        plt.close()
    else:
        return f


def input_to_device(inpt: Dict[str, Any], device: torch.device) -> None:
    """Moves all tensor values in a dictionary to the specified device.

    Args:
        inpt (Dict[str, Any]): Input dictionary containing tensors.
        device (torch.device): Target device.
    """
    for k, v in inpt.items():
        if isinstance(v, torch.Tensor):
            inpt[k] = v.to(device)


def load_sts_data(path: PathLike) -> Tuple[list, list, list]:
    """Loads STS data from a gzipped TSV file.

    Args:
        path (PathLike): Path to the dataset file.

    Returns:
        Tuple[list, list, list]: Train, dev, and test samples.
    """
    train_samples, dev_samples, test_samples = [], [], []
    with gzip.open(path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0
            sample = InputExample(
                texts=[row["sentence1"], row["sentence2"]], label=score
            )
            (
                dev_samples
                if row["split"] == "dev"
                else test_samples if row["split"] == "test" else train_samples
            ).append(sample)
    return train_samples, dev_samples, test_samples
