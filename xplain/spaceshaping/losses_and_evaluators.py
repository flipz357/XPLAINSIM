import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable

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


class DistilLoss(nn.Module):
    """
    Parameter-free loss module to distill metrics, decompose output space

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of features / metric scores. 1 will be added as residual
    :param feature_dim: Dimension of a feature
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.MSELoss()
    :param sim_fct: Optional: Custom similarity function. If not set, uses Manhatten Sim
     Example::
        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=[0.3, 0.2, 0.9]),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=[0.4, 0.1, 0.2])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.DistilLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3, feature_dim=16)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 feature_dims: list[int],
                 bias_inits: np.array = None,
                 loss_fct: Callable = nn.MSELoss(),
                 sim_fct: Callable = util.co_sim): #dist_sim
        super(DistilLoss, self).__init__()
        self.model = model
        self.sentence_embedding_dimension = sentence_embedding_dimension

        self.loss_fct = loss_fct
        self.sim_fct = sim_fct
        self.feature_dims = feature_dims
        self.num_labels = len(feature_dims)

        if bias_inits is None:
            biases = torch.ones(self.num_labels, requires_grad=False)
            self.score_bias = nn.Parameter(biases)
        else:
            
            biases = torch.tensor(bias_inits, requires_grad=False, dtype=torch.float32)
            self.score_bias = nn.Parameter(biases)

        self.score_bias.to(model._target_device)
    

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        """Compute the partitnioning loss sim(sub_embeddings) vs target metrics"""

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        
        sims = []
        start = 0
        for i in range(self.num_labels):
            
            # get dimensionality of feature
            feature_dim = self.feature_dims[i]

            # get two subembeddings
            stop = start + feature_dim
            rep_ax = rep_a[:, start:stop]
            rep_bx = rep_b[:, start:stop]

            # and compute their similariy
            sim = self.sim_fct(rep_ax, rep_bx)
            sims.append(sim)
            start += feature_dim

        # sims: (n_features x n_batch)
        # output: (n_batch x n_features)
        outputs = torch.stack(sims).T 
        outputs = self.score_bias * outputs
        
        if labels is not None:
            loss = self.loss_fct(outputs, labels)
            return loss 
        return reps, outputs
    
    def get_config_dict(self):
        return {'biases': self.score_bias}


class MultipleConsistencyLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        a learner and a teacher model. The loss computes a pairwise similarity matrix on the embeddings from the
        learner model A and for the teacher model B and tunes the Mean squared error (A - B)^2. I.e.,
        the learner is tuned to be consistend with the teacher.
    """
    def __init__(self, 
            model: SentenceTransformer, 
            teacher: SentenceTransformer,
            similarity_fct = stutil.cos_sim, 
            loss_fct: Callable = nn.MSELoss(),
            scale: float = 5.0):
        """
        :param model: SentenceTransformer model
        :param teacher: SentenceTransformer teacher, will be frozen
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :loss_fct: loss function
        :param scale: Output of similarity function is multiplied by scale value
        """
        super(MultipleConsistencyLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss_fct = loss_fct
        
        # freeze teacher
        self.teacher = teacher
        util.freeze_all_layers(self.teacher)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        
        teacher_reps = [self.teacher(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        teacher_embeddings_a = teacher_reps[0]
        teacher_embeddings_b = torch.cat(teacher_reps[1:])
        
        # intra model pairwise sims
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        
        # intra teacher pairwise sims
        teacher_scores = self.similarity_fct(teacher_embeddings_a, teacher_embeddings_b) * self.scale
        
        # (teacher_sim - model_sim)^2
        loss = self.loss_fct(scores, teacher_scores)
        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


class DistilConsistencyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, name: str = "", loss_model_distil = None, loss_model_consistency = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.loss_model_distil = loss_model_distil
        self.loss_model_consistency = loss_model_consistency

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "loss distill", "loss consistency", "biases"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        
        model.eval()
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        sum_mse_c = 0.0
        sum_mse_d = 0.0
        bxs = 0
        for _, batch in enumerate(self.dataloader):
            features, labels = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            labels = labels.to(model.device)
            
            
            with torch.no_grad():
                mse = self.loss_model_distil(features, labels=labels)
            sum_mse_d += mse
            with torch.no_grad():
                mse = self.loss_model_consistency(features, labels=labels)
            sum_mse_c += mse
            bxs += 1
         
        accuracy = (sum_mse_d + sum_mse_c) / bxs
        accuracy = 1 - accuracy
        biases = list(self.loss_model_distil.score_bias.detach().cpu().numpy())

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, sum_mse_d, sum_mse_c, biases])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow([epoch, steps, accuracy, sum_mse_d, sum_mse_c, biases])
        else:
            print(sum_mse_d, sum_mse_c)
        return accuracy

