import torch
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from xplain.spaceshaping.losses_and_evaluators import DistilLoss, MultipleConsistencyLoss, DistilConsistencyEvaluator
from xplain.spaceshaping import util

logger = logging.getLogger(__name__)

class PartitionedSentenceTransformer():

    def __init__(self,  feature_names: list, feature_dims:list, basemodelstring="all-MiniLM-L12-v2", 
                 device="cpu", tune_n_layers=2, batch_size=32, learning_rate=0.001,
                 epochs=2, warmup_steps=1000, eval_steps=200, save_path=None, write_csv=None):
        
        assert len(feature_names) == len(feature_dims)
        self.basemodelstring = basemodelstring
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.feature_dims = feature_dims
        self.device = device
        self.tune_n_layers = tune_n_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.save_path = save_path
        self.write_csv = write_csv
        self.init_models()

    def init_models(self):
        self.model = SentenceTransformer(self.basemodelstring, device=self.device)
        self.control = SentenceTransformer(self.basemodelstring, device=self.device)
        util.freeze_except_last_layers(self.model, self.tune_n_layers)
        util.freeze_all_layers(self.control)

    def train(self, train_examples, dev_examples):
        #self.init_models()
        train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
        dev_dataloader = torch.utils.data.DataLoader(dev_examples, shuffle=False, batch_size=self.batch_size)
        distill_loss = DistilLoss(self.model
                                        , sentence_embedding_dimension=self.model.get_sentence_embedding_dimension()
                                        , num_labels=self.n_features
                                        , feature_dim=self.feature_dims[0] # FIX
                                        , bias_inits=None)
        teacher_loss = MultipleConsistencyLoss(self.model, self.control)

        # init evaluator
        evaluator = DistilConsistencyEvaluator(dev_dataloader
                                                , loss_model_distil=distill_loss
                                                , loss_model_consistency=teacher_loss
                                                , write_csv = self.write_csv)

        #Tune the model
        self.model.fit(train_objectives=[(train_dataloader, teacher_loss), (train_dataloader, distill_loss)]
                                    , optimizer_params={'lr': self.learning_rate}
                                    , epochs=self.epochs
                                    , warmup_steps=self.warmup_steps
                                    , evaluator=evaluator
                                    , evaluation_steps=self.eval_steps
                                    , output_path=self.save_path
                                    , save_best_model=False)

    def explain_similarity(self, xsent, ysent):
        xsent_encoded = self.model.encode(xsent)
        ysent_encoded = self.model.encode(ysent)

        # cosine helper function
        def cosine_sim(mat1, mat2):
            prod = mat1 * mat2
            normf = lambda x: np.sqrt(np.sum(x**2, axis=1))
            normx, normy = normf(mat1), normf(mat2)
            return np.sum(prod, axis=1) / (normx * normy)
        
        total_features = sum(self.feature_dims)

        # global sbert sims
        simsglobal = cosine_sim(xsent_encoded, ysent_encoded)
        simsresidual = cosine_sim(xsent_encoded[:,total_features:], ysent_encoded[:,total_features:])

        metric_features = []
        #for i in range(0, dim*n, dim):
        #    metric_features.append((xsent_encoded[:,i:i+dim], ysent_encoded[:,i:i+dim]))
        curr = 0
        for dim in self.feature_dims:
            metric_features.append((xsent_encoded[:,curr:curr+dim], ysent_encoded[:,curr:curr+dim]))
            curr += dim

        metric_sims = []
        for i in range(len(self.feature_names)):
            xfea = metric_features[i][0]
            yfea = metric_features[i][1]
            simfea = cosine_sim(xfea, yfea)
            metric_sims.append(simfea)

        metric_sims = np.array(metric_sims)# * biases[:,np.newaxis]
        metric_sims = metric_sims.T

        preds = np.concatenate((simsglobal[:,np.newaxis], metric_sims, simsresidual[:,np.newaxis]), axis=1)
        verbose = []
        features = ["global"] + self.feature_names + ["residual"]
        for i, x in enumerate(xsent):
            sims = preds[i]
            jl = dict(zip(features, sims))
            jl["sent_a"] = x
            jl["sent_b"] = ysent[i]
            verbose.append(jl)
        return verbose

