import torch


# helpers

def interpolate_reference_embedding(embedding: torch.tensor, N: int):
    assert embedding.shape[0] == 2
    s = 1 / N
    device = embedding.device
    x, r = embedding[0], embedding[-1].unsqueeze(0)
    a = torch.arange(1, 0, -s).unsqueeze(1).unsqueeze(1).to(device)
    g = r + a * (x - r)
    g = torch.cat([g, r])
    return g

def repeat_reference_input(inpt: torch.tensor, N: int):
    x, r = inpt[0].unsqueeze(0), inpt[-1].unsqueeze(0)
    d_repeat = (N,) + (1,) * (len(x.shape) - 1)
    x = x.repeat(d_repeat)
    return torch.cat([x, r])


# roberta 

def roberta_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        return (g,) + inpt[1:]
    return hook

 
# mpnet

def mpnet_reshaping_hook(N: int):
    def hook(model, inpt):

        if len(inpt) > 2 and inpt[2] is not None:
            pos_bias = inpt[2]
            pos_bias = torch.cat(
                [pos_bias[0:1].repeat(N, 1, 1, 1), pos_bias[1:2]],
                dim=0
            )
        else:
            pos_bias = None

        if len(inpt) > 3:
            p = repeat_reference_input(inpt[3], N=N)
            return (inpt[0], inpt[1], pos_bias, p)

        return (inpt[0], inpt[1], pos_bias)

    return hook

def mpnet_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt):

        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)

        hidden_states = g
        attention_mask = inpt[1]

        # expand position bias if present
        if len(inpt) > 2 and inpt[2] is not None:
            pos_bias = inpt[2]
            pos_bias = torch.cat(
                [pos_bias[0:1].repeat(N, 1, 1, 1), pos_bias[1:2]],
                dim=0
            )
        else:
            pos_bias = None

        if len(inpt) > 3:
            p = repeat_reference_input(inpt[3], N=N)
            return (hidden_states, attention_mask, pos_bias, p)

        return (hidden_states, attention_mask, pos_bias)

    return hook


# gte

def gte_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        # expanding rope embeddings inplace
        with torch.no_grad():
            inpt[2][0].data = torch.cat([inpt[2][0][0:1].repeat(N, 1, 1, 1), inpt[2][0][1:2]])
            inpt[2][1].data = torch.cat([inpt[2][1][0:1].repeat(N, 1, 1, 1), inpt[2][1][1:2]])
        return (g,) + inpt[1:]
    return hook
