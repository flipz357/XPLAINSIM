import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, Any
from os import PathLike
import torch
import gzip
import csv
import pyemd
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


def trim_attributions_and_tokens(
    matrix: torch.Tensor,
    tokens_a: list,
    tokens_b: list,
    trim_start: int = 1,
    trim_end: int = 0,
) -> Tuple[torch.Tensor, list, list]:
    """Trims an attribution matrix and corresponding token lists.

    Args:
        matrix (torch.Tensor): Input attribution matrix.
        tokens_a (list): List of tokens corresponding to rows.
        tokens_b (list): List of tokens corresponding to columns.
        trim_start (int, optional): Number of tokens to trim from the start. Defaults to 1.
        trim_end (int, optional): Number of tokens to trim from the end. Defaults to 0.

    Returns:
        Tuple[torch.Tensor, list, list]: Trimmed matrix and token lists.

    Raises:
        ValueError: If trimming removes all tokens.
    """
    if trim_start + trim_end >= len(tokens_a) or trim_start + trim_end >= len(tokens_b):
        raise ValueError("Trimming exceeds the available number of tokens.")

    trimmed_matrix = matrix[
        trim_start : matrix.shape[0] - trim_end,
        trim_start : matrix.shape[1] - trim_end,
    ]

    trimmed_tokens_a = tokens_a[trim_start : len(tokens_a) - trim_end]
    trimmed_tokens_b = tokens_b[trim_start : len(tokens_b) - trim_end]

    return trimmed_matrix, trimmed_tokens_a, trimmed_tokens_b


def simple_align(attributions_matrix: torch.Tensor) -> np.ndarray:
    """Computes a simple sparcification alignment method through logical and between row wise and column wise maximum.

    Args:
        attributions_matrix (torch.Tensor): Attribution matrix to be postprocessed

    Returns:
        np.ndarray: Postprocessed attributions matrix (now sparsified)
    """
    numpy_attributions = attributions_matrix.numpy()
    row_maximum_attributions = assign_one_to_max(numpy_attributions, row_wise=True)
    col_maximum_attributions = assign_one_to_max(numpy_attributions, column_wise=True)
    return np.logical_and(row_maximum_attributions, col_maximum_attributions).astype(
        int
    )


def assign_one_to_max(
    numpy_attributions: np.ndarray, row_wise: bool = False, column_wise: bool = False
) -> np.ndarray:
    """Assigns 1 to the maximum value(s) in each row or column of a matrix.

    Args:
        matrix (np.ndarray): Input matrix.
        row_wise (bool, optional): If True, assigns 1 to max values row-wise. Defaults to False.
        column_wise (bool, optional): If True, assigns 1 to max values column-wise. Defaults to False.

    Returns:
        np.ndarray: Matrix with ones assigned to max values and zeros elsewhere.

    Raises:
        ValueError: If neither row_wise nor column_wise is set.
    """
    matrix = np.array(numpy_attributions)
    matched = np.zeros_like(numpy_attributions)

    if row_wise:
        max_indices = np.argmax(matrix, axis=1)
        matched[np.arange(matrix.shape[0]), max_indices] = 1
    elif column_wise:
        max_indices = np.argmax(matrix, axis=0)
        matched[max_indices, np.arange(matrix.shape[1])] = 1
    else:
        raise ValueError("Either row_wise or column_wise must be set to True.")

    return matched


def wasserstein_align(
    attributions_matrix: torch.Tensor, threshold: float = 0.029
) -> np.ndarray:
    """Computes Wasserstein alignment based on attribution flow.

    Args:
        attributions_matrix (torch.Tensor): Attribution matrix to be postprocessed
        threshold (float, optional): Threshold for binary alignment. Defaults to 0.029.

    Returns:
        np.ndarray: Postprocessed attributions matrix (now sparsified)
    """
    numpy_attributions = attributions_matrix.numpy()
    _, flow_matrix = attribs2emd_with_flow(numpy_attributions)
    return get_alignment_from_flow_cost(np.array(flow_matrix), threshold)


def get_token_weights(numpy_attributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes normalized token weights for left and right tokens.

    Args:
        numpy_attributions (np.ndarray): Attribution matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized token weights.
    """
    token_weights_left = np.ones(numpy_attributions.shape[0])
    token_weights_right = np.ones(numpy_attributions.shape[1])
    token_weights_left = np.concatenate(
        (np.zeros(numpy_attributions.shape[1]), token_weights_left)
    )
    token_weights_right = np.concatenate(
        (token_weights_right, np.zeros(numpy_attributions.shape[0]))
    )
    return token_weights_left / sum(token_weights_left), token_weights_right / sum(
        token_weights_right
    )


def get_alignment_from_flow_cost(
    flow_matrix: np.ndarray, threshold: float
) -> np.ndarray:
    """Transforms flow matrix into a binary alignment matrix.

    Args:
        flow_matrix (np.ndarray): The flow matrix.
        threshold (float): Threshold for binarization.

    Returns:
        np.ndarray: Binary alignment matrix.
    """
    binary_alignments = np.zeros(flow_matrix.shape)
    binary_alignments[flow_matrix >= threshold] = 1
    return binary_alignments


def pad_attribution_matrix(numpy_attributions: np.ndarray) -> np.ndarray:
    """Pads an attribution matrix.

    Args:
        numpy_attributions (np.ndarray): Input attribution matrix.

    Returns:
        np.ndarray: Padded attribution matrix.
    """
    dim = np.sum(numpy_attributions.shape)
    padded_attribution_matrix = np.zeros((dim, dim))
    for i in range(numpy_attributions.shape[0]):
        l = i + numpy_attributions.shape[1]
        for j in range(numpy_attributions.shape[1]):
            padded_attribution_matrix[l, j] = numpy_attributions[i, j]
    return padded_attribution_matrix


def attribs2cost(attribution_matrix: np.ndarray) -> np.ndarray:
    """Converts a similarity matrix to a cost matrix.

    Args:
        attribution_matrix (np.ndarray): Attribution matrix.

    Returns:
        np.ndarray: Cost matrix.
    """
    return 1 - 1 / (1 + np.exp(-attribution_matrix))


def attribs2emd_with_flow(attribution_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes EMD distance and flow matrix from an attribution matrix.

    Args:
        attribution_matrix (np.ndarray): Attribution matrix.

    Returns:
        Tuple[float, np.ndarray]: EMD distance and flow matrix.
    """
    token_weights_left, token_weights_right = get_token_weights(attribution_matrix)
    padded_attribution_matrix = pad_attribution_matrix(attribution_matrix)
    cost_matrix = attribs2cost(padded_attribution_matrix)
    distance, flow_matrix = pyemd.emd_with_flow(
        token_weights_left, token_weights_right, cost_matrix
    )
    return (
        distance,
        np.array(flow_matrix)[
            attribution_matrix.shape[1] :, : attribution_matrix.shape[1]
        ],
    )


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
