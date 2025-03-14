# Explaining Similarity

## Space Paritioning Example

Here's a very simple example for training and inferring with a custom model.

Needed: A training target. For every input text pair, a list with numbers. These numbers can be fine interpterable measurements. They are then used to structure the embedding space. In this example, we would like to build a model that reflects superficial semantic similarity in one part of its embedding, and deep semantic similarity in the other.

```python
from scipy.stats import pearsonr
from xplain.spaceshaping import PartitionedSentenceTransformer
from sentence_transformers import InputExample
from datasets import load_dataset

# let's first load a toy train dataset of sentence pairs
ds = load_dataset("mteb/stsbenchmark-sts")
some_pairs = list(zip([dic["sentence1"] for dic in ds["train"]], [dic["sentence2"] for dic in ds["train"]]))

# dev dataset of sentence pairs
some_pairs_dev = list(zip([dic["sentence1"] for dic in ds["validation"]], [dic["sentence2"] for dic in ds["validation"]]))

# let's build our target metrics that should be reflected within the embedding space
def bow_sim(x1, x2):
	x1 = set(x1.split())
	x2 = set(x2.split())
	inter = x1.intersection(x2)
	union = x1.union(x2)
	return len(inter) / len(union)

target = [[bow_sim(x1, x2)] for x1, x2 in some_pairs]
some_examples = [InputExample(texts=[x1, x2], label=target[i]) for (i, (x1, x2)) in enumerate(some_pairs)]

target_dev = [[bow_sim(x1, x2)] for x1, x2 in some_pairs_dev]
some_examples_dev = [InputExample(texts=[x1, x2], label=target_dev[i]) for (i, (x1, x2)) in enumerate(some_pairs_dev)]

# init model
pt = PartitionedSentenceTransformer(feature_names=["bow"], feature_dims=[32])
json = pt.explain_similarity([x for x, y in some_pairs_dev], [y for x, y in some_pairs_dev])

# eval correlation to custom metric before training
print(pearsonr([x.label[0] for x in some_examples_dev], [dic["bow"] for dic in json]))

# print a toy example before training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))

# train
pt.train(some_examples, some_examples_dev)

# eval correlation to custom metric after train
json = pt.explain_similarity([x for x, y in some_pairs_dev], [y for x, y in some_pairs_dev])
print(pearsonr([x.label[0] for x in some_examples_dev],[dic["bow"] for dic in json]))

print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
# print a toy example after training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
```

## To obtain attributions for an off-the-shelf transformer

```python
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
    A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```
