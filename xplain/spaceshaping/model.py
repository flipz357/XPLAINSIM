import torch
import numpy as np
import logging
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from xplain.spaceshaping.losses_and_evaluators import (
        PartitionLoss,
        ConsistencyLoss,
        CombinedLoss,
        MultiLossEvaluator
        )
from xplain.spaceshaping import util

logger = logging.getLogger(__name__)

class PartitionedSentenceTransformer(SentenceTransformer):

    def __init__(self,  feature_names: list[str], feature_dims: list[int], 
            base_model_uri: str = "all-MiniLM-L12-v2", 
            device: str = "cpu", tune_n_layers: str = 2):
        
        super().__init__(base_model_uri, device=device)

        assert len(feature_names) == len(feature_dims)
        assert sum(feature_dims) <= self.get_sentence_embedding_dimension()

        self.feature_names = feature_names
        self.feature_dims = feature_dims
        self.n_features = len(feature_names)

        util.freeze_except_last_layers(self, tune_n_layers)

    # ------------------------------------------------------------------
    # Loss Construction
    # ------------------------------------------------------------------

    def _build_control_model(self):
        """Create frozen teacher model."""
        control = SentenceTransformer(
            self._first_module().auto_model.name_or_path,
            device=self.device,
        )
        util.freeze_all_layers(control)
        return control

    def _build_losses(self, use_consistency: bool = True) -> Dict[str, torch.nn.Module]:
        """
        Build training losses.
        """

        losses = {}

        # Structured partition loss
        losses["partition"] = PartitionLoss(
            model=self,
            mode="metric",
            feature_dims=self.feature_dims,
            similarity="cosine",
        )

        # Optional teacher consistency
        if use_consistency:
            teacher = self._build_control_model()
            losses["consistency"] = ConsistencyLoss(
                model=self,
                teacher=teacher,
                mode="batch",
                similarity="cosine",
            )

        return losses

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    


    def train_model(
        self,
        train_examples,
        dev_examples,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 1,
        warmup_steps: int = 1000,
        eval_steps: int = 200,
        save_path: str = None,
        write_csv: bool = True,
        use_consistency: bool = True):
        """
        Train the partitioned model.
        """
        train_dl = torch.utils.data.DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size)

        dev_dl = torch.utils.data.DataLoader(
            dev_examples,
            shuffle=False,
            batch_size=batch_size)

        losses = self._build_losses(use_consistency=use_consistency)

        combined_loss = CombinedLoss(self, losses)

        evaluator = MultiLossEvaluator(
            dataloader=dev_dl,
            losses=losses,
            write_csv=write_csv)

        self.fit(
            train_objectives=[(train_dl, combined_loss)],
            optimizer_params={"lr": lr},
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=eval_steps,
            output_path=save_path,
            save_best_model=False)

    # ------------------------------------------------------------------
    # Embedding Partition Utilities
    # ------------------------------------------------------------------

    def split_embedding(self, embeddings: torch.Tensor, include_residual: bool = True) -> Dict[str, torch.Tensor]:
        """
        Split embeddings into feature subspaces + residual.
        """

        features = {}
        start = 0

        for name, dim in zip(self.feature_names, self.feature_dims):
            stop = start + dim
            features[name] = embeddings[:, start:stop]
            start = stop
        if include_residual:
            features["residual"] = embeddings[:, start:]
        return features

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_similarity(self, sent_a: List[str], sent_b: List[str]):
        """
        Return feature-wise similarity explanations.
        """

        emb_a = self.encode(sent_a)
        emb_b = self.encode(sent_b)

        parts_a = self.split_embedding(emb_a)
        parts_b = self.split_embedding(emb_b)

        def cosine_sim(x, y):
            return np.sum(x * y, axis=1) / (
                np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
            )

        explanations = []

        for i in range(len(sent_a)):

            explanation = {
                "sent_a": sent_a[i],
                "sent_b": sent_b[i],
            }

            # Feature-level similarities
            for name in self.feature_names:
                explanation[name] = cosine_sim(
                    parts_a[name][i:i+1],
                    parts_b[name][i:i+1],
                )[0]

            # Global similarity
            explanation["global"] = cosine_sim(
                emb_a[i:i+1],
                emb_b[i:i+1],
            )[0]

            explanations.append(explanation)

        return explanations
