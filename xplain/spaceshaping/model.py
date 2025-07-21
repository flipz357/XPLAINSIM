import torch
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from xplain.spaceshaping.losses_and_evaluators import DistilLossMetric, DistilLossDirect, MultipleConsistencyLossPaired, MultipleConsistencyLossDirect, DistilConsistencyEvaluator
from xplain.spaceshaping import util

logger = logging.getLogger(__name__)

class PartitionedSentenceTransformer():

    def __init__(self,  feature_names: list, feature_dims: list, base_model_uri="all-MiniLM-L12-v2", 
                 device="cpu", tune_n_layers=2, batch_size=32, learning_rate=0.001,
                 epochs=2, warmup_steps=1000, eval_steps=200, save_path=None, write_csv=None, sim_fct=util.co_sim):
        
        assert len(feature_names) == len(feature_dims)
        self.base_model_uri = base_model_uri
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
        self.sim_fct = sim_fct
        self.init_models()

    def init_models(self):
        self.model = SentenceTransformer(self.base_model_uri, device=self.device)
        self.control = SentenceTransformer(self.base_model_uri, device=self.device)
        util.freeze_except_last_layers(self.model, self.tune_n_layers)
        util.freeze_all_layers(self.control)

    def train(self, train_examples, dev_examples):
         
        train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
        dev_dataloader = torch.utils.data.DataLoader(dev_examples, shuffle=False, batch_size=self.batch_size)
        distill_loss = DistilLossMetric(self.model
                                        , sentence_embedding_dimension=self.model.get_sentence_embedding_dimension()
                                        , feature_dims=self.feature_dims
                                        , bias_inits=None
                                        , sim_fct = self.sim_fct)
        teacher_loss = MultipleConsistencyLossPaired(self.model, self.control)

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
    
    def train_direct(self, train_examples, dev_examples):
        assert [x == 1 for x in self.feature_dims]

        train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
        dev_dataloader = torch.utils.data.DataLoader(dev_examples, shuffle=False, batch_size=self.batch_size)
        distill_loss = DistilLossDirect(self.model
                                        , sentence_embedding_dimension=self.model.get_sentence_embedding_dimension()
                                        , feature_dims=self.feature_dims
                                        , bias_inits=None)

        teacher_loss = MultipleConsistencyLossDirect(self.model, self.control)

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

    def encode(self, sents):
        return self.model.encode(sents)
    
    def encode_features(self, sents):
        encoded = self.encode(sents)
        features = {}
        curr = 0
        for i, fea_name in enumerate(self.feature_names):
            dim = self.feature_dims[i]
            features[fea_name] = encoded[:, curr:curr+dim]
            curr += dim
        features["residual"] = encoded[:, curr:]
        return features
    
    def explain_similarity(self, xsent, ysent):
        feax = self.encode_features(xsent)
        feay = self.encode_features(ysent)
        
        # cosine helper function
        def cosine_sim(mat1, mat2):
            prod = mat1 * mat2
            normf = lambda x: np.sqrt(np.sum(x**2, axis=1))
            normx, normy = normf(mat1), normf(mat2)
            return np.sum(prod, axis=1) / (normx * normy)
        
        data = {}
        for fea_name in self.feature_names:
            x = feax[fea_name]
            y = feay[fea_name]
            data[fea_name] = cosine_sim(x, y)
        
        xglob = np.concatenate([feax[fn] for fn in self.feature_names + ["residual"]], axis=1)
        yglob = np.concatenate([feay[fn] for fn in self.feature_names + ["residual"]], axis=1)
        data["global"] = cosine_sim(xglob, yglob)
        explanations = []
        for i, sx in enumerate(xsent):
            sy = ysent[i]
            explanation = {}
            explanation["sent_a"] = sx
            explanation["sent_b"] = sy
            for fea_name in self.feature_names + ["global"]:
                explanation[fea_name] = data[fea_name][i]
            explanations.append(explanation)
        return explanations

