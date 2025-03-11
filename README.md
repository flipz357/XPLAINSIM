# Explaining Similarity

## Space Paritioning Example

Here's a very simple example for training and inferring with a custom model.

Needed: A training target. For every input text pair, a list with numbers. These numbers can be fine interpterable measurements. They are then used to structure the embedding space. In this example, we would like to build a model that reflects superficial semantic similarity in one part of its embedding, and deep semantic similarity in the other.

```python
from scipy.stats import pearsonr
from xplain.spaceshaping import PartitionedSentenceTransformer
from sentence_transformers import InputExample
from datasets import load_dataset

# let's first load a toy dataset of sentence pairs
ds = load_dataset("mteb/stsbenchmark-sts")
some_pairs = list(zip([dic["sentence1"] for dic in ds["train"]], [dic["sentence2"] for dic in ds["train"]]))

# let's build our target metrics that should be reflected within the embedding space
def bow_sim(x1, x2):
	x1 = set(x1.split())
	x2 = set(x2.split())
	inter = x1.intersection(x2)
	union = x1.union(x2)
	return len(inter) / len(union)

target = [[bow_sim(x1, x2)] for x1, x2 in some_pairs]
some_examples = [InputExample(texts=[x1, x2], label=target[i]) for (i, (x1, x2)) in enumerate(some_pairs)]

# init model
pt = PartitionedSentenceTransformer(feature_names=["bow"], feature_dims=[100])
json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])

# eval correlation to custom metric before training
print(pearsonr([x.label[0] for x in some_examples],[dic["bow"] for dic in json]))

# print a toy example before training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))

# train
pt.train(some_examples, some_examples)

# eval correlation to custom metric after train
json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])
print(pearsonr([x.label[0] for x in some_examples],[dic["bow"] for dic in json]))

print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
# print a toy example after training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
```
