# Explaining Similarity

## Overview of Repository / Table of Contents


- [**Installation**](#requirements)
- [**Attributions**](#attribution)
- [**Space Shaping**](#space-shaping)
    - [**Idea**](#space-shaping-idea)
    - [**Toy Example**](#space-shaping-toy)
- [**FAQ**](#faq)
- [**Citation**](#citation)

## Installation<a id="requirements"></a>

## To obtain attributions for an off-the-shelf transformer<a id="attribution"></a>

```python
from sentence_transformers.models import Pooling
from xplain.attribution import utils, ReferenceTransformer, XSMPNet
transformer = ReferenceTransformer('sentence-transformers/all-mpnet-base-v2')
pooling = Pooling(transformer.get_word_embedding_dimension())
model = XSMPNet(modules=[transformer, pooling])
model.reset_attribution()
model.init_attribution_to_layer(idx=10, N_steps=50)
texta = 'The dog runs after the kitten in the yard.'
textb = 'Outside in the garden the cat is chased by the dog.'
A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```

## Space partitioning<a id="space-shaping"></a>

### Idea<a id="space-shaping-idea"></a>

The idea is as follows: You have a bunch of interpreatble measures (`my_metrics`) and wish that these are reflected within sub-embeddings (`features`), while not disturbing the overall similarity too much.

```python
from sentence_transformers import InputExample
from xplain.spaceshaping import PartitionedSentenceTransformer

# need some documents pairs, don't need to be paraphrases, or similar, just some documents
list_with_strings, other_list_with_strings = ["abc",....], ["xyz",...]
examples = []

# compute the training/partitioning target
for x, y in zip(list_with_strings, other_list_with_strings):
	similarities = []
	for metric in my_metrics:
		similarities.append(metric.score(x, y))
	examples.append(InputExample(texts=[x, y], label=similarities))

# instantiate model and train, here we use 16 dimensions to express each metric
pt = PartitionedSentenceTransformer(feature_names=[metric.name for metric in my_metrics], 
                                    feature_dims=[16]*len(my_metrics))
pt.train(examples)
```

### Space Paritioning Example<a id="space-shaping-toy"></a>

Here's a very simple example for training and inferring with a custom model.

Needed: A training target. For every input text pair, a list with numbers. These numbers can be fine interpterable measurements. They are then used to structure the embedding space. In this example, we would like to build a model that reflects superficial semantic similarity in one part of its embedding, similarity of named entities in another, and "deep" semantic similarity in the other. Concretely, we parition three features

1. bag-of-words: Learns to reflect bag-of-words distance
2. named entity similarity: Learns to reflect similarity of named entities
3. (Not explicitly trained): Residual features for capturing the semantic similarity that makes for "the rest"

Note that this is only a toy code, and the training happens on very little data, however, the feature paritioning will already have some effect.

```python
from scipy.stats import pearsonr
from xplain.spaceshaping import PartitionedSentenceTransformer
from sentence_transformers import InputExample
from datasets import load_dataset
import spacy
nlp=spacy.load("en_core_web_sm")

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

def ner_sim(x1, x2, docs1, docs2):
	x1_ner = " ".join([ne.text for ne in docs1.ents])
	x2_ner = " ".join([ne.text for ne in docs2.ents])
	if not x1_ner and not x2_ner:
		return 1.0
	return bow_sim(x1_ner, x2_ner)

docs1, docs2 = [nlp(x) for x, _ in some_pairs], [nlp(y) for _, y in some_pairs]
target = [[bow_sim(x1, x2), ner_sim(x1, x2, docs1, docs2)] for x1, x2 in some_pairs]
some_examples = [InputExample(texts=[x1, x2], label=target[i]) for (i, (x1, x2)) in enumerate(some_pairs)]

docs1_dev, docs2_dev = [nlp(x) for x, _ in some_pairs_dev], [nlp(y) for _, y in some_pairs_dev]
target_dev = [[bow_sim(x1, x2), ner_sim(x1, x2, docs1_dev, docs2_dev)] for x1, x2 in some_pairs_dev]
some_examples_dev = [InputExample(texts=[x1, x2], label=target_dev[i]) for (i, (x1, x2)) in enumerate(some_pairs_dev)]

# init model
pt = PartitionedSentenceTransformer(feature_names=["bow", "ner"], feature_dims=[32, 32])
json = pt.explain_similarity([x for x, y in some_pairs_dev], [y for x, y in some_pairs_dev])

# eval correlation to custom metric before training
print(pearsonr([x.label[0] for x in some_examples_dev], [dic["bow"] for dic in json]))
print(pearsonr([x.label[1] for x in some_examples_dev], [dic["ner"] for dic in json]))

# print a toy example before training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))

# train
pt.train(some_examples, some_examples_dev)

# eval correlation to custom metric after train
json = pt.explain_similarity([x for x, y in some_pairs_dev], [y for x, y in some_pairs_dev])
print(pearsonr([x.label[0] for x in some_examples_dev],[dic["bow"] for dic in json]))
print(pearsonr([x.label[1] for x in some_examples_dev], [dic["ner"] for dic in json]))

# print a toy example after training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
```

## FAQ<a id="faq"></a>

## Citation<a id="citation"></a>
