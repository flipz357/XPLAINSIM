import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch
import string

try:
    import pyemd
except ModuleNotFoundError:
    print("For using alignment to post-process attributions, please install pyemd")
    

def xlm_roberta_tokenizer_merge_subtokens(tokens):
    """
    Merges sub-tokens into their original tokens.
    
    Args:
        tokens (list of str): List of sub-tokens.
    
    Returns:
        merged_tokens (list of str): List of merged tokens.
        index_map (list of list of int): Mapping from merged tokens to their original sub-token indices.
    """
    merged_tokens = []
    index_map = []
    current_token = ""
    current_indices = []

    for i, token in enumerate(tokens):
        if token in ['CLS', 'EOS'] or token in string.punctuation:  # Special tokens and punctuation
            if current_token:  # Add any current token being built
                merged_tokens.append(current_token)
                index_map.append(current_indices)
            merged_tokens.append(token)  # Add the special token or punctuation
            index_map.append([i])
            current_token = ""
            current_indices = []
        elif token.startswith('▁'):  # Start of a new token
            if current_token:  # If there's an existing token, add it to the list
                merged_tokens.append(current_token)
                index_map.append(current_indices)
            current_token = token[1:]  # Remove the '▁' prefix
            current_indices = [i]
        else:
            current_token += token
            current_indices.append(i)

    # Add the last token if it exists
    if current_token:
        merged_tokens.append(current_token)
        index_map.append(current_indices)

    return merged_tokens, index_map

def mpnet_tokenizer_merge_subtokens(tokens):
    """
    Merges sub-tokens into their original tokens.  Used for: "all-mpnet-base-v2"
    
    Args:
        tokens (list of str): List of sub-tokens.
    
    Returns:
        merged_tokens (list of str): List of merged tokens.
        index_map (list of list of int): Mapping from merged tokens to their original sub-token indices.
    """
    merged_tokens = []
    index_map = []
    current_token = ""
    current_indices = []

    for i, token in enumerate(tokens):
        if token in ['CLS', 'EOS'] or token in string.punctuation:  # Special tokens and punctuation
            if current_token:  # Add any current token being built
                merged_tokens.append(current_token)
                index_map.append(current_indices)
            merged_tokens.append(token)  # Add the special token or punctuation
            index_map.append([i])
            current_token = ""
            current_indices = []
        elif token.startswith('##'): # extra subtoken
            current_token += token[2:] # remove the extra '##'
            current_indices.append(i)
        else:
            if current_token:  # If there's an existing token, add it to the list
                merged_tokens.append(current_token)
                index_map.append(current_indices)
            current_token = token
            current_indices = [i]

    # Add the last token if it exists
    if current_token:
        merged_tokens.append(current_token)
        index_map.append(current_indices)

    return merged_tokens, index_map

def adjust_matrix_to_full_tokens(A_align, index_map_a, index_map_b, aggregation):
            """
            Adjusts the alignment matrix based on the merged token indices.
            
            Args:
                A_align (torch.Tensor): The original alignment matrix between TOKENS sub-tokens.
                index_map_a (list of list of int): Mapping from merged tokens to their original sub-token indices for the first sequence.
                index_map_b (list of list of int): Mapping from merged tokens to their original sub-token indices for the second sequence.
                aggregation (str): The method for aggregating alignment scores.
            
            Returns:
                new_matrix (torch.Tensor): The adjusted alignment matrix between the original tokens.
            """
            new_matrix = torch.zeros((len(index_map_a), len(index_map_b)))

            for i, indices_a in enumerate(index_map_a):
                for j, indices_b in enumerate(index_map_b):
                    values = A_align[torch.tensor(indices_a)[:, None], torch.tensor(indices_b)]
                    
                    if aggregation == 'mean':
                        new_matrix[i, j] = values.mean()
                    elif aggregation == 'min':
                        new_matrix[i, j] = values.min()
                    elif aggregation == 'max':
                        new_matrix[i, j] = values.max()
                    elif aggregation == 'sum':
                        new_matrix[i, j] = values.sum()
                    else:
                        raise ValueError("Invalid aggregation method. Choose from 'mean', 'min', 'max', 'sum'.")

            return new_matrix

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


def max_align(attributions_matrix: torch.Tensor) -> np.ndarray:
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


def flow_align(
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

