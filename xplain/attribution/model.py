"""Attribution models for token-level similarity explanation.

Provides ``ModelFactory`` for building attribution-ready sentence
transformers, and architecture-specific subclasses of ``XSTransformer``
that register interpolation hooks for integrated Jacobian computation.
"""

from sentence_transformers import models, SentenceTransformer
import torch
from torch import Tensor
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Union, List, Dict


from xplain.attribution.util import input_to_device
from xplain.attribution.postprocessing import (
    trim_attributions_and_tokens,
    max_align,
    flow_align,
    xlm_roberta_tokenizer_merge_subtokens,
    adjust_matrix_to_full_tokens,
    mpnet_tokenizer_merge_subtokens,
)
from xplain.attribution import hooks


class ModelFactory:
    """Factory for building attribution-ready sentence transformers."""

    def __init__(self):
        return None

    @staticmethod
    def show_options():
        """Return list of known model name strings."""
        return list(ModelFactory._get_model_reference_dict().keys())

    @staticmethod
    def _get_model_reference_dict():
        """Return mapping of model names to ``(class, HuggingFace URI)`` pairs."""
        dic = {
            "all-mpnet-base-v2": (XSMPNet, "sentence-transformers/all-mpnet-base-v2"),
            "xlm-roberta-base": (XSDefaultEncoder, "FacebookAI/xlm-roberta-base"),
            "multilingual-e5-base": (XSDefaultEncoder, "intfloat/multilingual-e5-base"),
            "paraphrase-multilingual-mpnet-base": (
                XSDefaultEncoder,
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ),
            "paraphrase-multilingual-MiniLM": (
                XSDefaultEncoder,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            "gte-multilingual-base": (XGTE, "Alibaba-NLP/gte-multilingual-base"),
            "sentence-transformers/all-mpnet-base-v2": (
                XSMPNet,
                "sentence-transformers/all-mpnet-base-v2",
            ),
            "FacebookAI/xlm-roberta-base": (
                XSDefaultEncoder,
                "FacebookAI/xlm-roberta-base",
            ),
            "intfloat/multilingual-e5-base": (
                XSDefaultEncoder,
                "intfloat/multilingual-e5-base",
            ),
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": (
                XSDefaultEncoder,
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
                XSDefaultEncoder,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            "Alibaba-NLP/gte-multilingual-base": (
                XGTE,
                "Alibaba-NLP/gte-multilingual-base",
            ),
        }
        return dic

    @staticmethod
    def build(modelname: str, layer_idx=-2, N_steps=50):
        """Build an attribution model by name.

        Args:
            modelname: Model identifier (see ``show_options()``).
                Unknown names fall back to ``XSDefaultEncoder``.
            layer_idx: Encoder layer to hook for interpolation
                (negative indices count from the end).
            N_steps: Number of interpolation steps for the
                integrated Jacobian.

        Returns:
            An initialised ``XSTransformer`` subclass with hooks
            registered and subtoken merging configured.
        """
        maybe_models = ModelFactory._get_model_reference_dict().get(
            modelname, (XSDefaultEncoder, modelname)
        )  # added default model support.
        modelclass, reference = maybe_models
        transformer = ReferenceTransformer(reference)
        pooling = models.Pooling(transformer.get_word_embedding_dimension())
        model = modelclass(
            modules=[transformer, pooling]
        )  # if model is unsupported it will error out in the next two lines
        model.reset_attribution()
        model.init_attribution_to_layer(idx=layer_idx, N_steps=N_steps)
        model.initialise_subtoken_to_tokens_method()
        return model


class ReferenceTransformer(models.Transformer):
    """Transformer that appends a reference input to every batch.

    The reference (a neutral token-id sequence) is concatenated to the
    batch before the forward pass so that downstream hooks can
    interpolate between the actual and reference embeddings.
    """

    def forward(self, features):

        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        s = input_ids.shape[1]
        device = input_ids.device
        # TODO: generalize to arbitrary tokenizer
        ref_ids = torch.IntTensor([[0] + [1] * (s - 2) + [2]]).to(device)
        features["input_ids"] = torch.cat([input_ids, ref_ids], dim=0)
        if input_ids.shape[0] > 1:
            ref_att = torch.ones((1, s)).int().to(device)
            features["attention_mask"] = torch.cat([attention_mask, ref_att], dim=0)

        return super().forward(features)

        # emb = features['token_embeddings']

        # att = features['attention_mask']
        # if att.shape[0] > 1:
        #     att = att[:-1]

        # features.update({
        #     'token_embeddings': emb,
        #     'attention_mask': att
        # })

        # return features

    @staticmethod
    def load(input_path: str):
        """Load a ReferenceTransformer from a saved directory."""
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return ReferenceTransformer(model_name_or_path=input_path, **config)


class XSTransformer(SentenceTransformer):
    """Base class for attribution-ready sentence transformers.

    Strips the appended reference from the output and stores it
    separately. Subclasses implement ``init_attribution_to_layer``
    to register architecture-specific interpolation hooks.
    """

    def forward(self, features: dict, **kwargs):
        """Forward pass; splits off the reference embedding."""
        features = super().forward(features, **kwargs)
        emb = features["sentence_embedding"]
        att = features["attention_mask"]
        features.update({"sentence_embedding": emb[:-1], "attention_mask": att[:-1]})
        features["reference"] = emb[-1]
        features["reference_att"] = att[-1]
        return features

    def init_attribution_to_layer(self, idx: int, N_steps: int):
        """Register interpolation hooks at the given layer. Subclass responsibility."""
        raise NotImplementedError()

    def reset_attribution(self):
        """Remove all registered interpolation hooks. Subclass responsibility."""
        raise NotImplementedError()

    def initialise_subtoken_to_tokens_method(self):
        """Set the subtoken-to-token merging function. Subclass responsibility."""
        raise NotImplementedError()

    def tokenize_text(self, text: str):
        """Tokenize text and wrap with CLS/EOS markers."""
        tokens = self[0].tokenizer.tokenize(text)
        tokens = [t[1:] if t[0] in ["Ġ", "Â"] else t for t in tokens]
        tokens = ["CLS"] + tokens + ["EOS"]
        return tokens

    def _compute_integrated_jacobian(
        self,
        embedding: torch.Tensor,
        intermediate: torch.Tensor,
        move_to_cpu: bool = True,
        verbose: bool = True,
    ):
        """Compute the integrated Jacobian of the embedding w.r.t. an intermediate.

        Iterates over embedding dimensions, accumulating per-dimension
        gradients, then averages over interpolation steps.

        Args:
            embedding: Sentence embedding tensor to differentiate.
            intermediate: Interpolated intermediate activations
                (captured by the hook).
            move_to_cpu: If True, move gradients to CPU after each step.
            verbose: If True, show a tqdm progress bar.

        Returns:
            Integrated Jacobian tensor, summed over interpolation steps.
        """
        D = embedding.shape[1]
        jacobi = []
        retain_graph = True
        for d in tqdm(range(D), disable=not verbose):
            if d == D - 1:
                retain_graph = False
            grads = torch.autograd.grad(
                list(embedding[:, d]), intermediate, retain_graph=retain_graph
            )[0].detach()
            if move_to_cpu:
                grads = grads.cpu()
            jacobi.append(grads)
        J = torch.stack(jacobi) / self.N_steps
        J = J[:, :-1, :, :].sum(dim=1)
        return J

    def _process(
        self,
        sent: str,
        intermediates_index: int,
        move_to_cpu: bool = False,
        verbose: bool = True,
        dims: Optional[Tuple] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Tensor:
        """Encode a single sentence and compute its integrated Jacobian.

        Args:
            sent: Input sentence.
            intermediates_index: Index into ``self.intermediates`` for the
                hook-captured activations.
            move_to_cpu: If True, move results to CPU.
            verbose: If True, show progress.
            dims: Optional ``(start, end)`` slice of embedding dimensions.
            device: Device to run on.

        Returns:
            Tuple of ``(embedding, delta, jacobian, seq_len,
            embed_dim, features)``.
        """

        inpt = self[0].tokenize([sent])
        input_to_device(inpt, device)
        features = self.forward(inpt, **kwargs)
        emb = features["sentence_embedding"]
        if dims is not None:
            emb = emb[:, dims[0] : dims[1]]
        interm = self.intermediates[intermediates_index]
        J = self._compute_integrated_jacobian(
            emb, interm, move_to_cpu=move_to_cpu, verbose=verbose
        )
        D, S, D = J.shape

        J = J.reshape((D, S * D))

        d = interm[0] - interm[-1]
        d = d.reshape(S * D, 1).detach()
        if move_to_cpu:
            d = d.cpu()
        return emb, d, J, S, D, features

    def explain_similarity(
        self,
        sent_a: Union[List[str], str],
        sent_b: Union[List[str], str],
        sim_measure: str = "cosine",
        return_lhs_terms: bool = False,
        move_to_cpu: bool = False,
        verbose: bool = True,
        compress_embedding_dim: bool = True,
        dims: Optional[Tuple] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Union[
        Tuple[Tensor, List[str], List[str]], List[Tuple[Tensor, List[str], List[str]]]
    ]:
        """Decompose embedding similarity into token-level attributions.

        For a pair of sentences, computes a matrix whose entries
        approximate each token pair's contribution to the overall
        similarity score.

        Args:
            sent_a: First sentence(s). String or list of strings.
            sent_b: Second sentence(s). Must match type of ``sent_a``.
            sim_measure: ``"cosine"`` or ``"dot"``.
            return_lhs_terms: If True, also return the scalar similarity
                and reference-based decomposition terms.
            move_to_cpu: If True, move all tensors to CPU.
            verbose: If True, show progress bars.
            compress_embedding_dim: If True, sum over embedding
                dimensions to produce a ``(Sa, Sb)`` matrix.
            dims: Optional ``(start, end)`` embedding dimension slice.
            device: Device to run on.

        Returns:
            ``(attribution_matrix, tokens_a, tokens_b)`` for string
            inputs. When ``return_lhs_terms`` is True, four additional
            scalars are appended. For list inputs, each element of the
            tuple is a list over sentence pairs.
        """

        if device is None:
            device = next(self.parameters()).device

        if sim_measure not in {"cosine", "dot"}:
            raise ValueError(
                f"invalid argument for sim_measure: {sim_measure}, must be cos or dot"
            )

        self.intermediates.clear()
        # device = self[0].auto_model.embeddings.word_embeddings.weight.device

        if isinstance(sent_a, list) and isinstance(sent_b, list):
            explanations = []
            for sa, sb in zip(sent_a, sent_b):
                output = self.explain_similarity(
                    sa,
                    sb,
                    sim_measure,
                    return_lhs_terms,
                    move_to_cpu,
                    verbose,
                    compress_embedding_dim,
                    dims,
                    device,
                )
                if not explanations:
                    for i, item in enumerate(output):
                        explanations.append([item])
                else:
                    for i, item in enumerate(output):
                        explanations[i].append(item)
            return explanations
        elif isinstance(sent_a, list) or isinstance(sent_b, list):
            raise ValueError("sent_a and sent_b must both be strings or both lists")

        self.intermediates.clear()
        # device = self[0].auto_model.embeddings.word_embeddings.weight.device

        emb_a, da, J_a, Sa, Da, features_a = self._process(
            sent_a,
            0,
            move_to_cpu=move_to_cpu,
            verbose=verbose,
            dims=dims,
            device=device,
        )
        emb_b, db, J_b, Sb, Db, features_b = self._process(
            sent_b,
            1,
            move_to_cpu=move_to_cpu,
            verbose=verbose,
            dims=dims,
            device=device,
        )

        J = torch.mm(J_a.T, J_b)
        da = da.repeat(1, Sb * Db)
        db = db.repeat(1, Sa * Da)
        if move_to_cpu:
            da = da.detach().cpu()
            db = db.detach().cpu()
            emb_a = emb_a.detach().cpu()
            emb_b = emb_b.detach().cpu()
            J = J.cpu()
        A = da * J * db.T
        if sim_measure == "cosine":
            A = A / torch.norm(emb_a[0]) / torch.norm(emb_b[0])
        A = A.reshape(Sa, Da, Sb, Db)
        if compress_embedding_dim:
            A = A.sum(dim=(1, 3))
        A = A.detach().cpu()

        tokens_a = self.tokenize_text(sent_a)
        tokens_b = self.tokenize_text(sent_b)

        if return_lhs_terms:
            ref_a = features_a["reference"]
            ref_b = features_b["reference"]
            if dims is not None:
                ref_a = ref_a[dims[0] : dims[1]]
                ref_b = ref_b[dims[0] : dims[1]]
            if move_to_cpu:
                ref_a = ref_a.detach().cpu()
                ref_b = ref_b.detach().cpu()
            if sim_measure == "cosine":
                score = torch.cosine_similarity(
                    emb_a[0].unsqueeze(0), emb_b[0].unsqueeze(0)
                ).item()
                ref_emb_a = torch.cosine_similarity(
                    emb_a[0].unsqueeze(0), ref_b.unsqueeze(0)
                ).item()
                ref_emb_b = torch.cosine_similarity(
                    emb_b[0].unsqueeze(0), ref_a.unsqueeze(0)
                ).item()
                ref_ref = torch.cosine_similarity(
                    ref_a.unsqueeze(0), ref_b.unsqueeze(0)
                ).item()
            elif sim_measure == "dot":
                score = torch.dot(emb_a[0], emb_b[0]).item()
                ref_emb_a = torch.dot(emb_a[0], ref_b).item()
                ref_emb_b = torch.dot(emb_b[0], ref_a).item()
                ref_ref = torch.dot(ref_a, ref_b).item()
            return A, tokens_a, tokens_b, score, ref_emb_a, ref_emb_b, ref_ref
        else:
            return A, tokens_a, tokens_b

    def explain_by_decomposition(
        self, text_a: str, text_b: str, normalize: bool = False
    ):
        """Compute a simple token interaction matrix via dot-product decomposition.

        Unlike ``explain_similarity``, this does not use integrated
        gradients — it directly multiplies token embeddings.

        Args:
            text_a: First sentence.
            text_b: Second sentence.
            normalize: If True, normalize each token embedding by its sum.

        Returns:
            Tuple of ``(interaction_matrix, tokens_a, tokens_b)``.
        """

        device = self[0].auto_model.embeddings.word_embeddings.weight.device

        inpt_a = self[0].tokenize([text_a])
        input_to_device(inpt_a, device)
        emb_a = self.forward(inpt_a)["token_embeddings"][0]
        if normalize:
            emb_a = emb_a / torch.sum(emb_a)

        inpt_b = self[0].tokenize([text_b])
        input_to_device(inpt_b, device)
        emb_b = self.forward(inpt_b)["token_embeddings"][0]
        if normalize:
            emb_b = emb_b / torch.sum(emb_b)

        A = torch.mm(emb_a, emb_b.t()).detach().cpu()
        A = A / emb_a.shape[0] / emb_b.shape[0]

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        return A, tokens_a, tokens_b

    def token_sim_mat(self, text_a: str, text_b: str, layer: int, device: torch.device):
        """Compute a raw token-token similarity matrix at a given layer.

        Args:
            text_a: First sentence.
            text_b: Second sentence.
            layer: Encoder layer index to extract embeddings from.
            device: Device to run on.

        Returns:
            Tuple of ``(similarity_matrix, tokens_a, tokens_b)``.
        """

        self[0].auto_model.config.output_hidden_states = True

        inpt_a = self[0].tokenize([text_a])
        inpt_b = self[0].tokenize([text_b])
        input_to_device(inpt_a, device)
        input_to_device(inpt_b, device)

        with torch.no_grad():
            emb_a = self[0].forward(inpt_a)["all_layer_embeddings"][layer][0]
            emb_b = self[0].forward(inpt_b)["all_layer_embeddings"][layer][0]

        A = torch.mm(emb_a, emb_b.t()).detach().cpu()

        self[0].auto_model.config.output_hidden_states = False

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        return A, tokens_a, tokens_b

    def score(self, texts: Tuple[str]):
        """Compute the dot-product similarity between two texts.

        Args:
            texts: Tuple of two sentence strings.

        Returns:
            Scalar similarity score.
        """
        self.eval()
        with torch.no_grad():
            inputs = [self[0].tokenize([t]) for t in texts]
            for inpt in inputs:
                input_to_device(inpt, self.device)
            embeddings = [self.forward(inpt)["sentence_embedding"] for inpt in inputs]
            s = torch.dot(embeddings[0][0], embeddings[1][0]).cpu().item()
            del embeddings
            torch.cuda.empty_cache()
        return s

    def postprocess_attributions(
        self,
        matrix: torch.Tensor,
        tokens_a: list,
        tokens_b: list,
        subtokens_to_tokens: Optional[bool] = True,
        subtokens_aggregation_method: Optional[str] = "mean",
        sparsification_method: Optional[str] = None,
        flowalign_sparsify_threshold: float = 0.029,
        trim_starting_tokens: int = 1,
        trim_ending_tokens: int = 1,
    ) -> Tuple[np.ndarray, list, list]:
        """Postprocess an attribution matrix by trimming, merging, and sparsifying.

        Args:
            matrix: The attribution matrix to be processed.
            tokens_a: Tokens corresponding to the rows of the matrix.
            tokens_b: Tokens corresponding to the columns of the matrix.
            subtokens_to_tokens: If True, merge sub-tokens into full
                tokens.
            subtokens_aggregation_method: Aggregation when merging
                sub-tokens (``"mean"``, ``"min"``, ``"max"``, ``"sum"``).
            sparsification_method: ``"FlowAlign"``, ``"MaxAlign"``, or
                None.
            flowalign_sparsify_threshold: Threshold for FlowAlign
                binarisation.
            trim_starting_tokens: Tokens to trim from the start of each
                token list.
            trim_ending_tokens: Tokens to trim from the end of each
                token list.

        Returns:
            Tuple of ``(processed_matrix, tokens_a, tokens_b)``.
        """
        matrix, tokens_a, tokens_b = trim_attributions_and_tokens(
            matrix=matrix,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            trim_start=trim_starting_tokens,
            trim_end=trim_ending_tokens,
        )

        if subtokens_to_tokens:
            matrix, tokens_a, tokens_b = self.merge_subtokens_and_adjust_matrix(
                matrix, tokens_a, tokens_b, subtokens_aggregation_method
            )

        if sparsification_method == "FlowAlign":
            matrix = flow_align(
                attributions_matrix=matrix, threshold=flowalign_sparsify_threshold
            )
        elif sparsification_method == "MaxAlign":
            matrix = max_align(attributions_matrix=matrix)

        return matrix, tokens_a, tokens_b

    def merge_subtokens_and_adjust_matrix(
        self, A_align, tokens_a, tokens_b, aggregation="sum"
    ):
        """Merge sub-tokens into full tokens and adjust the alignment matrix.

        Args:
            A_align: Alignment matrix between sub-tokens.
            tokens_a: Sub-tokens from the first sequence.
            tokens_b: Sub-tokens from the second sequence.
            aggregation: Method for aggregating scores over merged
                sub-tokens (``"sum"``, ``"mean"``, ``"min"``, ``"max"``).

        Returns:
            Tuple of ``(adjusted_matrix, merged_tokens_a,
            merged_tokens_b)``.
        """

        # Merge sub-tokens to form original tokens
        tokens_a_merged, index_map_a = self.token_to_subtoken_method(tokens_a)
        tokens_b_merged, index_map_b = self.token_to_subtoken_method(tokens_b)

        # Adjust the alignment matrix
        A_align_new = adjust_matrix_to_full_tokens(
            A_align, index_map_a, index_map_b, aggregation
        )

        return A_align_new, tokens_a_merged, tokens_b_merged


class XSDefaultEncoder(XSTransformer):
    """Attribution support for RoBERTa / XLM-RoBERTa style encoders."""

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")

        num_layers = len(self[0].auto_model.encoder.layer)
        if idx < 0:  # We inverse the index (to allow -2, -3 initialisation)
            assert (
                abs(idx) < num_layers
            ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx  # its always negative
        assert (
            idx < num_layers
        ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
        try:
            self.N_steps = N_steps
            self.intermediates = []
            self.hook = (
                self[0]
                .auto_model.encoder.layer[idx]
                .register_forward_pre_hook(
                    hooks.roberta_interpolation_hook(
                        N=N_steps, outputs=self.intermediates
                    )
                )
            )
        except AttributeError:
            raise AttributeError("The encoder model is not supported")

    def reset_attribution(self):
        if hasattr(self, "hook"):
            self.hook.remove()
            self.hook = None
        else:
            print("No hook has been registered.")

    def initialise_subtoken_to_tokens_method(self):
        if not hasattr(self, "token_to_subtoken_method"):
            self.token_to_subtoken_method = xlm_roberta_tokenizer_merge_subtokens


class XSMPNet(XSTransformer):
    """Attribution support for MPNet encoders."""

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.interpolation_hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(self[0].auto_model.encoder.layer)
        if idx < 0:  # We inverse the index (to allow -2, -3 initialisation)
            assert (
                abs(idx) < num_layers
            ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx  # its always negative
        assert (
            idx < num_layers
        ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
        try:
            self.N_steps = N_steps
            self.intermediates = []
            self.interpolation_hook = (
                self[0]
                .auto_model.encoder.layer[idx]
                .register_forward_pre_hook(
                    hooks.mpnet_interpolation_hook(
                        N=N_steps, outputs=self.intermediates
                    )
                )
            )
            self.reshaping_hooks = []
            for l in range(idx + 1, len(self[0].auto_model.encoder.layer)):
                handle = (
                    self[0]
                    .auto_model.encoder.layer[l]
                    .register_forward_pre_hook(hooks.mpnet_reshaping_hook(N=N_steps))
                )
                self.reshaping_hooks.append(handle)
        except AttributeError:
            raise AttributeError("The encoder model is not supported")

    def reset_attribution(self):
        if hasattr(self, "interpolation_hook"):
            self.interpolation_hook.remove()
            del self.interpolation_hook
            for hook in self.reshaping_hooks:
                hook.remove()
            del self.reshaping_hooks

    def initialise_subtoken_to_tokens_method(self):
        if not hasattr(self, "token_to_subtoken_method"):
            self.token_to_subtoken_method = mpnet_tokenizer_merge_subtokens


class XGTE(XSTransformer):
    """Attribution support for GTE encoders (with RoPE)."""

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(self[0].auto_model.encoder.layer)
        if idx < 0:  # We inverse the index (to allow -2, -3 initialisation)
            assert (
                abs(idx) < num_layers
            ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx  # its always negative
        assert (
            idx < num_layers
        ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
        try:
            self.N_steps = N_steps
            self.intermediates = []
            self.hook = (
                self[0]
                .auto_model.encoder.layer[idx]
                .register_forward_pre_hook(
                    hooks.gte_interpolation_hook(N=N_steps, outputs=self.intermediates)
                )
            )
        except AttributeError:
            raise AttributeError("The encoder model is not supported")

    def reset_attribution(self):
        if hasattr(self, "hook"):
            self.hook.remove()
            self.hook = None
        else:
            print("No hook has been registered.")

    def initialise_subtoken_to_tokens_method(self):
        if not hasattr(self, "token_to_subtoken_method"):
            self.token_to_subtoken_method = xlm_roberta_tokenizer_merge_subtokens


class XJina(XSTransformer):
    """Attribution support for Jina encoders."""

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(self[0].auto_model.roberta.encoder.layers)
        if idx < 0:  # We inverse the index (to allow -2, -3 initialisation)
            assert (
                abs(idx) < num_layers
            ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx  # its always negative
        assert (
            idx < num_layers
        ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
        try:
            self.N_steps = N_steps
            self.intermediates = []
            self.hook = (
                self[0]
                .auto_model.roberta.encoder.layers[idx]
                .register_forward_pre_hook(
                    hooks.roberta_interpolation_hook(
                        N=N_steps, outputs=self.intermediates
                    )
                )
            )
        except AttributeError:
            raise AttributeError("The encoder model is not supported")

    def reset_attribution(self):
        if hasattr(self, "hook"):
            self.hook.remove()
            self.hook = None
        else:
            print("No hook has been registered.")


class XGTR(XSTransformer):
    """Attribution support for GTR (T5-based) encoders."""

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(self[0].auto_model.encoder.block)
        if idx < 0:  # We inverse the index (to allow -2, -3 initialisation)
            assert (
                abs(idx) < num_layers
            ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx  # its always negative
        assert (
            idx < num_layers
        ), f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
        try:
            self.N_steps = N_steps
            self.intermediates = []
            self.hook = (
                self[0]
                .auto_model.encoder.block[idx]
                .register_forward_pre_hook(
                    hooks.roberta_interpolation_hook(
                        N=N_steps, outputs=self.intermediates
                    )
                )
            )
        except AttributeError:
            raise AttributeError("The encoder model is not supported")

    def reset_attribution(self):
        if hasattr(self, "hook"):
            self.hook.remove()
            self.hook = None
        else:
            print("No hook has been registered.")
