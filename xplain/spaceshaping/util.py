from torch import Tensor
import torch
import torch.nn.functional as F

def freeze_all_layers(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return None

def freeze_except_last_layers(model, n=2):
    
    layernames = list([name for name, _ in model.named_parameters() if "layer." in name])
    layerids = [name.split("layer.")[1].split(".")[0] for name in layernames]
    layerids = set([int(lid) for lid in layerids])
    layerids = sorted(list(layerids))
    lastn = layerids[-n:]
    lastn = ["layer." + str(lid) for lid in lastn]

    for name, param in model.named_parameters():
        lid = None
        if "layer." in name:
            lid = "layer." + name.split("layer.")[1].split(".")[0]
        if lid and lid in lastn:
            continue
        param.requires_grad = False
    
    return None

def dist_sim(reps1: Tensor, reps2: Tensor):
    """ based on manhatten distance """
    diff = torch.abs(reps1 - reps2)
    sim = 1.0 - torch.sum(diff, dim=1)
    return sim

def normed_dist_sim(reps1: Tensor, reps2: Tensor):
    """ based on manhatten distance """
    #reps1_norm = torch.sum(reps1 ** 2, dim=1)
    #reps2_norm = torch.sum(reps2 ** 2, dim=1)
    #reps1_norm = torch.sqrt(reps1_norm)
    #reps2_norm = torch.sqrt(reps2_norm)
    #reps1_norm = reps1.norm(p=2, dim=1, keepdim=True)
    
    reps1 = F.normalize(reps1, p=2, dim=1)
    reps2 = F.normalize(reps2, p=2, dim=1)
    diff = torch.abs(reps1 - reps2)
    sim = 1.0 - torch.sum(diff, dim=1)
    return sim


def prod_sim(reps1: Tensor, reps2: Tensor):
    """ dot product """
    diff = reps1 * reps2
    sim = torch.sum(diff, dim=1)
    return sim

def co_sim(reps1: Tensor, reps2: Tensor):
    """ cosinus similarity """
    sim = prod_sim(reps1, reps2)
    reps1_norm = torch.sum(reps1 ** 2, dim=1)
    reps2_norm = torch.sum(reps2 ** 2, dim=1)
    reps1_norm = torch.sqrt(reps1_norm)
    reps2_norm = torch.sqrt(reps2_norm)
    sim /= (reps1_norm * reps2_norm)
    return sim
