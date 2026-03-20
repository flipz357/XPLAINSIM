"""Partitioned sentence transformer with interpretable embedding subspaces."""

import torch
import logging
from typing import List, Dict, Optional, Union

from sentence_transformers import SentenceTransformer, InputExample
from xplain.spaceshaping.losses_and_evaluators import (
    PartitionLoss,
    ConsistencyLoss,
    CombinedLoss,
    MultiLossEvaluator,
)
from xplain.spaceshaping import util

logger = logging.getLogger(__name__)


class PartitionedSentenceTransformer(SentenceTransformer):
    """SentenceTransformer with feature-partitioned embedding space.

    The embedding vector is divided into predefined subspaces,
    each intended to encode a specific interpretable similarity aspect.
    These partitions can be trained using structured losses
    to align with external metrics.

    Args:
        feature_names: Names of interpretable embedding partitions.
        feature_dims: Dimensionality of each partition.
            ``sum(feature_dims)`` must be less than the embedding dimension.
        base_model_uri: Name or path of the underlying SentenceTransformer.
        similarity: Key into ``util.SIMILARITY_REGISTRY``.
        device: Device for model computation.
        tune_n_layers: Number of final transformer layers to keep trainable.
    """

    def __init__(
        self,
        feature_names: list[str],
        feature_dims: list[int],
        base_model_uri: str = "all-MiniLM-L12-v2",
        similarity: str = "cosine",
        device: str = "cpu",
        tune_n_layers: int = 2,
    ):

        super().__init__(base_model_uri, device=device)

        assert len(feature_names) == len(feature_dims)
        assert sum(feature_dims) < self.get_sentence_embedding_dimension()
        assert similarity in util.SIMILARITY_REGISTRY

        self.similarity_name = similarity
        self.similarity_fct = util.SIMILARITY_REGISTRY[similarity]
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
        """Build training losses.

        Args:
            use_consistency: If True, include a teacher-student consistency
                loss alongside the partition loss.

        Returns:
            Mapping of loss names to loss modules.
        """

        losses = {}

        # Structured partition loss
        losses["partition"] = PartitionLoss(
            model=self,
            mode="metric",
            similarity=self.similarity_name,
        )

        # Optional teacher consistency
        if use_consistency:
            teacher = self._build_control_model()
            losses["consistency"] = ConsistencyLoss(
                model=self,
                teacher=teacher,
                mode="batch",
                similarity=self.similarity_name,
            )

        return losses

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        train_examples: List[InputExample],
        dev_examples: Optional[List[InputExample]] = None,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 1,
        warmup_steps: int = 100,
        eval_steps: int = 200,
        save_path: str = None,
        write_csv: bool = True,
        use_consistency: bool = True,
    ):
        """Train the partitioned model.

        Uses structured partition loss and optional teacher consistency
        regularisation.

        Args:
            train_examples: Training examples (InputExample format).
            dev_examples: Development examples for evaluation. If None,
                no evaluation is performed.
            batch_size: Training batch size.
            lr: Learning rate.
            epochs: Number of training epochs.
            warmup_steps: Warmup steps for the scheduler.
            eval_steps: Evaluation interval (in steps).
            save_path: Directory to save the model.
            write_csv: Whether to log evaluation results to CSV.
            use_consistency: Whether to include teacher-student
                consistency loss.
        """

        train_dl = torch.utils.data.DataLoader(
            train_examples, shuffle=True, batch_size=batch_size
        )

        losses = self._build_losses(use_consistency=use_consistency)
        combined_loss = CombinedLoss(self, losses)

        evaluator = None
        if dev_examples is not None:
            dev_dl = torch.utils.data.DataLoader(
                dev_examples, shuffle=False, batch_size=batch_size
            )

            evaluator = MultiLossEvaluator(
                dataloader=dev_dl, losses=losses, write_csv=write_csv
            )

        self.fit(
            train_objectives=[(train_dl, combined_loss)],
            optimizer_params={"lr": lr},
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=eval_steps if evaluator else 0,
            output_path=save_path,
            save_best_model=False,
        )

    # ------------------------------------------------------------------
    # Embedding Partition Utilities
    # ------------------------------------------------------------------

    def split_embedding(
        self, embeddings: torch.Tensor, include_residual: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Split embeddings into predefined feature partitions.

        Args:
            embeddings: Tensor of shape ``(batch_size, embedding_dim)``.
            include_residual: If True, include remaining dimensions as
                ``"residual"``.

        Returns:
            Dictionary mapping feature names to subspace tensors.
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

    def explain_similarity(
        self, sent_a: Union[str, List[str]], sent_b: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Compute feature-wise similarity explanations.

        For each sentence pair, similarity is computed per feature
        partition and globally over the full embedding.

        Args:
            sent_a: First sentence(s). String or list of strings.
            sent_b: Second sentence(s). Must match type of ``sent_a``.

        Returns:
            One dictionary per sentence pair (or a single dict for
            string inputs) containing the original texts, per-partition
            similarities, and global similarity.
        """

        single_input = False
        if isinstance(sent_a, str) and isinstance(sent_b, str):
            single_input = True
            sent_a = [sent_a]
            sent_b = [sent_b]
        elif isinstance(sent_a, str) or isinstance(sent_b, str):
            raise ValueError("Input a and b must be both strings, or both lists")

        if len(sent_a) != len(sent_b):
            raise ValueError("sent_a and sent_b must have same length")
        emb_a = self.encode(sent_a, convert_to_tensor=True)
        emb_b = self.encode(sent_b, convert_to_tensor=True)

        parts_a = self.split_embedding(emb_a)
        parts_b = self.split_embedding(emb_b)

        # Precompute all similarities in batch
        feature_sims = {}
        for name in self.feature_names + ["residual"]:
            feature_sims[name] = self.similarity_fct(parts_a[name], parts_b[name])
        global_sims = self.similarity_fct(emb_a, emb_b)

        explanations = []

        for i in range(len(sent_a)):

            explanation = {
                "sent_a": sent_a[i],
                "sent_b": sent_b[i],
            }

            for name in self.feature_names + ["residual"]:
                explanation[name] = feature_sims[name][i].item()

            explanation["global"] = global_sims[i].item()

            explanations.append(explanation)

        return explanations[0] if single_input else explanations
