import torch
from nltk.corpus import stopwords


def join_words(tokens: list, join_char: str = '##'):
    word_indices = []
    words = []
    current_idxs = []
    current_str = ''
    for i, token in enumerate(tokens):
        current_idxs.append(i)
        if token.startswith(join_char):
            token = token[2:]
        current_str += token
        if i == len(tokens) - 1 or not tokens[i + 1].startswith(join_char):
            word_indices.append(current_idxs)
            words.append(current_str)
            current_idxs = []
            current_str = ''
    return words, word_indices


def compress_attributions(attributions: torch.tensor, indices: list, dim: int):
    compressed = []
    for idxs in indices:
        slice = torch.index_select(
            attributions, index=torch.tensor(idxs), dim=dim
            ).sum(dim=dim, keepdim=True)
        compressed.append(slice)
    return torch.cat(compressed, dim=dim)


def token_to_word_attributions(attributions:torch.tensor, tokens_a: list, tokens_b: list):
    words_a, word_idxs_a = join_words(tokens_a)
    words_b, word_idxs_b = join_words(tokens_b)
    A = attributions
    A = compress_attributions(A, word_idxs_a, dim=0)
    A = compress_attributions(A, word_idxs_b, dim=1)
    return A, words_a, words_b


def divide_stop_and_content_words(words: list, language: str = 'english'):
    stop_words = stopwords.words(language)
    stop_indices = []
    content_indices = []
    for index, word in enumerate(words):
        if word.lower() in stop_words:
            stop_indices.append(index)
        else:
            content_indices.append(index)
    return stop_indices, content_indices


def extract_stop_and_content_word_attribution(attributions: torch.tensor, stop_word_indices: list, content_word_indices: list, dim: int):
    selection_dim = dim
    pooling_dim = int(not bool(dim))
    if stop_word_indices:
        stop_attr = torch.index_select(attributions, index=torch.tensor(stop_word_indices), dim=selection_dim).sum(dim=pooling_dim)
    else:
        stop_attr = None
    if content_word_indices:
        content_attr = torch.index_select(attributions, index=torch.tensor(content_word_indices), dim=selection_dim).sum(dim=pooling_dim)
    else:
        content_attr = None
    return stop_attr, content_attr


def process_samples(samples: dict):
    stop_attributions = []
    content_attributions = []
    for i, s in enumerate(samples):
        try:
            query_tokens = s['query_tokens']
            passage_tokens = s['passage_tokens']
            attribution = s['attribution']
            word_attr, _, passage_words = token_to_word_attributions(attribution, query_tokens, passage_tokens)
            stop_indices, content_indices = divide_stop_and_content_words(passage_words)
            stop_attr, content_attr = extract_stop_and_content_word_attribution(word_attr, stop_indices, content_indices, dim=1)
            if stop_attr is not None:
                stop_attributions += list(stop_attr.numpy())
            if content_attr is not None:
                content_attributions += list(content_attr.numpy())
        except:
            print('error with sample',  i)
            continue
    return stop_attributions, content_attributions