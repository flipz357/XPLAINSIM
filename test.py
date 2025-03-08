def test_space_shaping():
    from scipy.stats import pearsonr
    from xplain.spaceshaping import PartitionedSentenceTransformer
    from sentence_transformers import InputExample
    pt = PartitionedSentenceTransformer(feature_names=["bow"], feature_dims=[100])
    from datasets import load_dataset
    ds = load_dataset("mteb/stsbenchmark-sts")
    some_pairs = list(zip([dic["sentence1"] for dic in ds["train"]], [dic["sentence2"] for dic in ds["train"]]))
    def bow_sim(x1, x2):
        x1 = set(x1)
        x2 = set(x2)
        inter = x1.intersection(x2)
        union = x1.union(x2)
        return len(inter) / len(union)
    target = [[bow_sim(x1, x2)] for x1, x2 in some_pairs]
    some_examples = [InputExample(texts=[x1, x2], label=target[i]) for (i, (x1, x2)) in enumerate(some_pairs)]
    json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])
    print(json)
    print(pearsonr([x.label[0] for x in some_examples],[dic["bow"] for dic in json]))
    pt.train(some_examples, some_examples)
    json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])
    print(json)
    print(pearsonr([x.label[0] for x in some_examples],[dic["bow"] for dic in json]))

def test_attribution():
    import torch
    from sentence_transformers.models import Pooling
    from xplain.attribution import utils, ReferenceTransformer, XSMPNet
    transformer = ReferenceTransformer('sentence-transformers/all-mpnet-base-v2')
    pooling = Pooling(transformer.get_word_embedding_dimension())
    model = XSMPNet(modules=[transformer, pooling])
    #model.to(torch.device('cuda:1'))
    model.reset_attribution()
    model.init_attribution_to_layer(idx=10, N_steps=50)
    texta = 'The dog runs after the kitten in the yard.'
    textb = 'Outside in the garden the cat is chased by the dog.'
    A, tokens_a, tokens_b = model.explain_similarity(
    texta, 
    textb, 
    move_to_cpu=True,
    sim_measure='cos')

#test_attribution()
test_space_shaping()
