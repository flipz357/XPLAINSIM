import torch
from torch import nn, Tensor
from typing import Iterable, Dict, List, Callable, Optional

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as stutil
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
import logging
import numpy as np
from xplain.spaceshaping import util
import os
import csv

logger = logging.getLogger(__name__)


class PartitionLoss(nn.Module):
    """
    Loss to decompose output space

    mode:
        - "metric": similarity between embedding partitions
        - "direct": direct regression on full embedding
    model: SentenceTransformer model
    feature_dim: Optional Dimension of a feature, if None then 1
    loss_fct: Optional Custom pytorch loss function. If not set, uses nn.MSELoss()
    sim_fct: Optional: Custom similarity function. If not set, uses Cosine Sim
    """
    def __init__(self,
                 model: SentenceTransformer,
                 mode: str,
                 feature_dims: Optional[List[int]] = None,
                 similarity: str = "cosine",
                 loss_fct: Callable = nn.MSELoss(),
                 use_bias: bool = False):
        
        super().__init__()

        assert mode in ["metric", "direct"]
        assert similarity in util.SIMILARITY_REGISTRY

        self.model = model
        self.mode = mode
        self.feature_dims = feature_dims
        self.similarity_name = similarity
        self.similarity_fct = util.SIMILARITY_REGISTRY[similarity]
        self.loss_fct = loss_fct
        self.use_bias = use_bias

        if self.mode == "metric":
            assert feature_dims is not None
            self.num_features = len(feature_dims)

            if use_bias:
                self.score_bias = nn.Parameter(torch.ones(self.num_features))
            else:
                self.register_parameter("score_bias", None)
    
    def forward(self, reps, labels=None, sentence_features=None):

        if labels is None:
            raise ValueError("PartitionLoss requires labels.")

        if self.mode == "metric":

            emb_a, emb_b = reps

            parts_a = self.model.split_embedding(emb_a, include_residual=False)
            parts_b = self.model.split_embedding(emb_b, include_residual=False)

            sims = [self.similarity_fct(parts_a[name], parts_b[name]) for name in self.model.feature_names]

            outputs = torch.stack(sims, dim=1)

            if self.use_bias:
                outputs = outputs * self.score_bias

            return self.loss_fct(outputs, labels)

        # direct mode
        emb = reps[0]
        k = labels.shape[1]
        return self.loss_fct(emb[:, :k], labels)

    def get_config_dict(self):
        return {
            "mode": self.mode,
            "feature_dims": self.feature_dims,
            "similarity": self.similarity_name,
            "loss_fct": self.loss_fct.__class__.__name__,
            "use_bias": self.use_bias,
        }


class ConsistencyLoss(nn.Module):
    """
        This loss aligns the overall similarities of a learner with thouse of a teacher.

        Modes: 
            "paired" accepts two lists where texts are parallel (i.e., similar).
            "batch" accepts one list with texts.
    """
    def __init__(self, 
                model: SentenceTransformer, 
                teacher: SentenceTransformer,
                mode: str = "paired",
                similarity: str = "cosine",
                loss_fct: Callable = nn.MSELoss(),
                scale: float = 5.0):
        
        super().__init__()
        
        assert mode in ["paired", "batch"]
        assert similarity in util.SIMILARITY_REGISTRY

        self.model = model
        self.teacher = teacher
        self.mode = mode
        self.similarity_name = similarity
        self.similarity_fct = util.SIMILARITY_REGISTRY[similarity]
        self.loss_fct = loss_fct
        self.scale = scale

        # freeze teacher
        util.freeze_all_layers(self.teacher)

    def forward(self, reps, labels=None, sentence_features=None):

        student_reps = reps 

        with torch.no_grad():
            teacher_reps = [self.teacher(sf)["sentence_embedding"] for sf in sentence_features]

        if self.mode == "paired":

            s_a = student_reps[0]
            s_b = torch.cat(student_reps[1:], dim=0)

            t_a = teacher_reps[0]
            t_b = torch.cat(teacher_reps[1:], dim=0)

            score_s = self.similarity_fct(s_a, s_b) * self.scale
            score_t = self.similarity_fct(t_a, t_b) * self.scale

            return self.loss_fct(score_s, score_t)

        # batch mode

        s_all = torch.cat(student_reps, dim=0)
        t_all = torch.cat(teacher_reps, dim=0)

        sim_s = self.similarity_fct(s_all, s_all) * self.scale
        sim_t = self.similarity_fct(t_all, t_all) * self.scale

        return self.loss_fct(sim_s, sim_t)

    
    def get_config_dict(self):
        return {
            "mode": self.mode,
            "similarity": self.similarity_name,
            "loss_fct": self.loss_fct.__class__.__name__,
            "scale": self.scale,
        }


class CombinedLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, losses: Dict[str, nn.Module]):
        super().__init__()
        self.model = model
        self.losses = losses

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):

        # Compute student embeddings ONCE
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]

        total_loss = 0.0

        for loss in self.losses.values():
            total_loss += loss(reps, labels, sentence_features=sentence_features)

        return total_loss


class MultiLossEvaluator(SentenceEvaluator):

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        losses: Dict[str, nn.Module],
        name: str = "",
        write_csv: bool = True,
    ):
        self.dataloader = dataloader
        self.losses = losses
        self.name = name
        self.write_csv = write_csv

        if name:
            name = "_" + name

        self.csv_file = f"evaluation{name}_results.csv"
        self.csv_headers = ["epoch", "steps", "score"] + list(losses.keys())

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):

        model.eval()
        self.dataloader.collate_fn = model.smart_batching_collate

        loss_sums = {name: 0.0 for name in self.losses}
        batches = 0

        for batch in self.dataloader:
            features, labels = batch

            # Move to device
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            labels = labels.to(model.device)

            with torch.no_grad():

                # Compute embeddings once
                reps = [
                    model(sf)["sentence_embedding"]
                    for sf in features
                ]

                # Evaluate each loss using shared embeddings
                for name, loss_model in self.losses.items():

                    loss_val = loss_model(
                        reps,
                        labels=labels,
                        sentence_features=features
                    )

                    loss_sums[name] += loss_val.item()

            batches += 1

        # Average losses
        for name in loss_sums:
            loss_sums[name] /= batches

        total_loss = sum(loss_sums.values())
        score = 1 - total_loss

        # CSV writing
        if output_path and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            write_header = not os.path.isfile(csv_path)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")

                if write_header:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [epoch, steps, score] +
                    [loss_sums[name] for name in self.losses]
                )

        return score
