"""Forward pre-hooks for integrated-gradient interpolation.

Each hook factory returns a closure that can be registered via
``register_forward_pre_hook`` on a transformer layer. The hooks
interpolate between the actual input embedding and a reference
embedding, which is required for the integrated Jacobian computation
in the attribution module.
"""

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def interpolate_reference_embedding(embedding: torch.tensor, N: int):
    """Create N linearly-spaced interpolations from embedding to reference.

    Args:
        embedding: Tensor of shape ``(2, seq_len, hidden_dim)`` where
            ``embedding[0]`` is the actual input and ``embedding[-1]``
            is the reference.
        N: Number of interpolation steps.

    Returns:
        Tensor of shape ``(N + 1, seq_len, hidden_dim)`` — the N
        interpolated points followed by the reference.
    """
    assert embedding.shape[0] == 2
    s = 1 / N
    device = embedding.device
    x, r = embedding[0], embedding[-1].unsqueeze(0)
    a = torch.arange(1, 0, -s).unsqueeze(1).unsqueeze(1).to(device)
    g = r + a * (x - r)
    g = torch.cat([g, r])
    return g


def repeat_reference_input(inpt: torch.tensor, N: int):
    """Repeat the actual input N times and append the reference.

    Used to broadcast non-interpolated inputs (e.g. position ids) so
    they match the batch dimension produced by interpolation.

    Args:
        inpt: Tensor of shape ``(2, ...)`` — actual input and reference.
        N: Number of repeats for the actual input.

    Returns:
        Tensor of shape ``(N + 1, ...)``.
    """
    x, r = inpt[0].unsqueeze(0), inpt[-1].unsqueeze(0)
    d_repeat = (N,) + (1,) * (len(x.shape) - 1)
    x = x.repeat(d_repeat)
    return torch.cat([x, r])


# ---------------------------------------------------------------------------
# RoBERTa / XLM-RoBERTa
# ---------------------------------------------------------------------------


def roberta_interpolation_hook(N: int, outputs: list):
    """Hook factory for RoBERTa-style encoders.

    Interpolates hidden states and stores them in ``outputs`` for
    later Jacobian computation.
    """

    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        return (g,) + inpt[1:]

    return hook


# ---------------------------------------------------------------------------
# MPNet
# ---------------------------------------------------------------------------


def mpnet_reshaping_hook(N: int):
    """Hook factory for MPNet layers after the interpolation layer.

    Expands position bias and extra positional inputs to match the
    interpolated batch size, without performing interpolation itself.
    """

    def hook(model, inpt):

        if len(inpt) > 2 and inpt[2] is not None:
            pos_bias = inpt[2]
            pos_bias = torch.cat(
                [pos_bias[0:1].repeat(N, 1, 1, 1), pos_bias[1:2]], dim=0
            )
        else:
            pos_bias = None

        if len(inpt) > 3:
            p = repeat_reference_input(inpt[3], N=N)
            return (inpt[0], inpt[1], pos_bias, p)

        return (inpt[0], inpt[1], pos_bias)

    return hook


def mpnet_interpolation_hook(N: int, outputs: list):
    """Hook factory for the MPNet interpolation layer.

    Interpolates hidden states, stores them in ``outputs``, and
    expands position bias and extra positional inputs to match.
    """

    def hook(model, inpt):

        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)

        hidden_states = g
        attention_mask = inpt[1]

        # expand position bias if present
        if len(inpt) > 2 and inpt[2] is not None:
            pos_bias = inpt[2]
            pos_bias = torch.cat(
                [pos_bias[0:1].repeat(N, 1, 1, 1), pos_bias[1:2]], dim=0
            )
        else:
            pos_bias = None

        if len(inpt) > 3:
            p = repeat_reference_input(inpt[3], N=N)
            return (hidden_states, attention_mask, pos_bias, p)

        return (hidden_states, attention_mask, pos_bias)

    return hook


# ---------------------------------------------------------------------------
# GTE (with RoPE)
# ---------------------------------------------------------------------------


def gte_interpolation_hook(N: int, outputs: list):
    """Hook factory for GTE encoders that use rotary position embeddings.

    Interpolates hidden states, stores them in ``outputs``, and
    expands RoPE embeddings in-place to match the interpolated batch.
    """

    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        # expanding rope embeddings inplace
        with torch.no_grad():
            inpt[2][0].data = torch.cat(
                [inpt[2][0][0:1].repeat(N, 1, 1, 1), inpt[2][0][1:2]]
            )
            inpt[2][1].data = torch.cat(
                [inpt[2][1][0:1].repeat(N, 1, 1, 1), inpt[2][1][1:2]]
            )
        return (g,) + inpt[1:]

    return hook
