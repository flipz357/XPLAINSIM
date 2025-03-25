from sentence_transformers import models, SentenceTransformer
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from xplain.attribution.utils import input_to_device
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

    def __init__(self):
        return None

    @staticmethod
    def show_options():
        return list(ModelFactory._get_model_reference_dict().keys())

    @staticmethod
    def _get_model_reference_dict():
        dic = {
            "all-mpnet-base-v2": (XSMPNet, "sentence-transformers/all-mpnet-base-v2"),
            "xlm-roberta-base": (XSDefaultEncoder, "FacebookAI/xlm-roberta-base"),
            "multilingual-e5-base" : (XSDefaultEncoder, "intfloat/multilingual-e5-base"),
            "paraphrase-multilingual-mpnet-base" : (XSDefaultEncoder, "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            "paraphrase-multilingual-MiniLM" : (XSDefaultEncoder, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            "gte-multilingual-base": (XGTE, "Alibaba-NLP/gte-multilingual-base"),
            "sentence-transformers/all-mpnet-base-v2": (XSMPNet, "sentence-transformers/all-mpnet-base-v2"),
            "FacebookAI/xlm-roberta-base": (XSDefaultEncoder, "FacebookAI/xlm-roberta-base"),
            "intfloat/multilingual-e5-base" : (XSDefaultEncoder, "intfloat/multilingual-e5-base"),
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" : (XSDefaultEncoder, "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" : (XSDefaultEncoder, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            "Alibaba-NLP/gte-multilingual-base": (XGTE, "Alibaba-NLP/gte-multilingual-base")
        }
        return dic

    @staticmethod
    def build(modelname: str, layer_idx=-2, N_steps=50):
        maybe_models = ModelFactory._get_model_reference_dict().get(modelname, (XSDefaultEncoder, modelname)) # added default model support. 
        modelclass, reference = maybe_models
        transformer = ReferenceTransformer(reference)
        pooling = models.Pooling(transformer.get_word_embedding_dimension())
        model = modelclass(modules=[transformer, pooling]) # if model is unsupported it will error out in the next two lines
        model.reset_attribution()
        model.init_attribution_to_layer(idx=layer_idx, N_steps=N_steps) 
        model.initialise_subtoken_to_tokens_method()
        return model


class ReferenceTransformer(models.Transformer):
    """adds reference to batch but does not subtract its embeddings after forward"""

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

    def forward(self, features: dict, **kwargs):
        features = super().forward(features, **kwargs)
        emb = features["sentence_embedding"]
        att = features["attention_mask"]
        features.update({"sentence_embedding": emb[:-1], "attention_mask": att[:-1]})
        features["reference"] = emb[-1]
        features["reference_att"] = att[-1]
        return features

    def init_attribution_to_layer(self, idx: int, N_steps: int):
        raise NotImplementedError()

    def reset_attribution(self):
        raise NotImplementedError()
    
    def initialise_subtoken_to_tokens_method(self):
        raise NotImplementedError()

    def tokenize_text(self, text: str):
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

    def explain_similarity(
        self,
        text_a: str,
        text_b: str,
        sim_measure: str = "cos",
        return_lhs_terms: bool = False,
        move_to_cpu: bool = False,
        verbose: bool = True,
        compress_embedding_dim: bool = True,
        dims: Optional[Tuple] = None,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):

        assert sim_measure in [
            "cos",
            "dot",
        ], f"invalid argument for sim_measure: {sim_measure}"

        self.intermediates.clear()
        # device = self[0].auto_model.embeddings.word_embeddings.weight.device

        # TODO: this should be a method
        assert sim_measure in [
            "cos",
            "dot",
        ], f"invalid argument for sim_measure: {sim_measure}"

        self.intermediates.clear()
        # device = self[0].auto_model.embeddings.word_embeddings.weight.device

        # TODO: this should be a method
        inpt_a = self[0].tokenize([text_a])
        input_to_device(inpt_a, device)
        features_a = self.forward(inpt_a, **kwargs)
        emb_a = features_a["sentence_embedding"]
        if dims is not None:
            emb_a = emb_a[:, dims[0] : dims[1]]
        interm_a = self.intermediates[0]
        J_a = self._compute_integrated_jacobian(
            emb_a, interm_a, move_to_cpu=move_to_cpu, verbose=verbose
        )
        D, Sa, Da = J_a.shape
        J_a = J_a.reshape((D, Sa * Da))

        da = interm_a[0] - interm_a[-1]
        da = da.reshape(Sa * Da, 1).detach()

        inpt_b = self[0].tokenize([text_b])
        input_to_device(inpt_b, device)
        features_b = self.forward(inpt_b, **kwargs)
        emb_b = features_b["sentence_embedding"]
        if dims is not None:
            emb_b = emb_b[:, dims[0] : dims[1]]
        interm_b = self.intermediates[1]
        J_b = self._compute_integrated_jacobian(
            emb_b, interm_b, move_to_cpu=move_to_cpu, verbose=verbose
        )
        _, Sb, Db = J_b.shape
        J_b = J_b.reshape((D, Sb * Db))

        db = interm_b[0] - interm_b[-1]
        db = db.reshape(Sb * Db, 1).detach()

        J = torch.mm(J_a.T, J_b)
        da = da.repeat(1, Sb * Db)
        db = db.repeat(1, Sa * Da)
        if move_to_cpu:
            da = da.detach().cpu()
            db = db.detach().cpu()
            emb_a = emb_a.detach().cpu()
            emb_b = emb_b.detach().cpu()
        A = da * J * db.T
        if sim_measure == "cos":
            A = A / torch.norm(emb_a[0]) / torch.norm(emb_b[0])
        A = A.reshape(Sa, Da, Sb, Db)
        if compress_embedding_dim:
            A = A.sum(dim=(1, 3))
        A = A.detach().cpu()

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        if return_lhs_terms:
            ref_a = features_a["reference"]
            ref_b = features_b["reference"]
            if dims is not None:
                ref_a = ref_a[dims[0] : dims[1]]
                ref_b = ref_b[dims[0] : dims[1]]
            if move_to_cpu:
                ref_a = ref_a.detach().cpu()
                ref_b = ref_b.detach().cpu()
            if sim_measure == "cos":
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
        trim_ending_tokens: int = 1
    ) -> Tuple[np.ndarray, list, list]:
        """
        Postprocesses an attribution matrix and its associated tokens by trimming 
        unwanted tokens and optionally applying a sparsification method.

        Args:
            matrix (torch.Tensor): The attribution matrix to be processed.
            tokens_a (list): List of tokens corresponding to the rows of the matrix.
            tokens_b (list): List of tokens corresponding to the columns of the matrix.
            sparsification_method (Optional[str], optional): The sparsification method 
                to apply. Can be "FlowAlign", "MaxAlign", or None. Defaults to None.
            wasserstein_sparsify_threshold (float, optional): Threshold used in the 
                FlowAlign method. Defaults to 0.029.
            trim_starting_tokens (int, optional): Number of tokens to trim from the start 
                of each token list. Defaults to 1.
            trim_ending_tokens (int, optional): Number of tokens to trim from the end 
                of each token list. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, list, list]: A tuple containing the processed attribution 
            matrix and the trimmed token lists (tokens_a, tokens_b).
        """
        matrix, tokens_a, tokens_b = trim_attributions_and_tokens(
            matrix=matrix,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            trim_start=trim_starting_tokens,
            trim_end=trim_ending_tokens,
        )
        
        if subtokens_to_tokens:
            matrix, tokens_a, tokens_b = self.merge_subtokens_and_adjust_matrix(matrix, tokens_a, tokens_b, subtokens_aggregation_method)

        if sparsification_method == "FlowAlign":
            matrix = flow_align(
                attributions_matrix=matrix,
                threshold=flowalign_sparsify_threshold
            )
        elif sparsification_method == "MaxAlign":
            matrix = max_align(attributions_matrix=matrix)

        return matrix, tokens_a, tokens_b
    
    def merge_subtokens_and_adjust_matrix(self, A_align, tokens_a, tokens_b, aggregation='sum'):
        """
        Merges sub-tokens into their original tokens and adjusts the alignment matrix accordingly.
        
        Args:
            A_align (torch.Tensor): The original alignment matrix between sub-tokens.
            tokens_a (list of str): List of sub-tokens from the first sequence.
            tokens_b (list of str): List of sub-tokens from the second sequence.
            aggregation (str, optional): The method for aggregating alignment scores. 
                                        Options are 'mean', 'min', and 'max'. Default is 'mean'.

        Returns:
            A_align_new (torch.Tensor): The adjusted alignment matrix between the original tokens.
            tokens_a_merged (list of str): The merged tokens from the first sequence.
            tokens_b_merged (list of str): The merged tokens from the second sequence.
        """

        # Merge sub-tokens to form original tokens
        tokens_a_merged, index_map_a = self.token_to_subtoken_method(tokens_a)
        tokens_b_merged, index_map_b = self.token_to_subtoken_method(tokens_b)

        # Adjust the alignment matrix
        A_align_new = adjust_matrix_to_full_tokens(A_align, index_map_a, index_map_b, aggregation)

        return A_align_new, tokens_a_merged, tokens_b_merged

        

class XSDefaultEncoder(XSTransformer):

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        
        num_layers = len(
            self[0].auto_model.encoder.layer
        )
        if idx < 0: # We inverse the index (to allow -2, -3 initialisation)
            assert abs(idx) < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx # its always negative
        assert idx < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
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

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.interpolation_hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(
            self[0].auto_model.encoder.layer
        )
        if idx < 0: # We inverse the index (to allow -2, -3 initialisation)
            assert abs(idx) < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx # its always negative
        assert idx < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
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

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(
            self[0].auto_model.encoder.layer
        )
        if idx < 0: # We inverse the index (to allow -2, -3 initialisation)
            assert abs(idx) < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx # its always negative
        assert idx < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
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

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(
            self[0].auto_model.roberta.encoder.layers
        )
        if idx < 0: # We inverse the index (to allow -2, -3 initialisation)
            assert abs(idx) < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx # its always negative
        assert idx < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
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

    def init_attribution_to_layer(self, idx: int, N_steps: int):

        if hasattr(self, "hook") and self.hook is not None:
            raise AttributeError("a hook is already registered")
        num_layers = len(
            self[0].auto_model.encoder.block
        )
        if idx < 0: # We inverse the index (to allow -2, -3 initialisation)
            assert abs(idx) < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
            idx = num_layers + idx # its always negative
        assert idx < num_layers, f"the model does not have a layer {idx}. The model has {num_layers} available layers. "
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
