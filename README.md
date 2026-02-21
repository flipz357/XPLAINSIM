 # XPLAINSIM: A Toolkit for Explaining Text Similarity

A package for explaining and exploring text similarity with diverse methods.

## Overview of Repository / Table of Contents


- [Installation](#requirements)
- [**Attributions**](#attribution)
	- [Example Code](#attributions-example)
	- [Expansion: Subtokens-to-Tokens](#attributions-subtokens-to-tokens)  
	- [Expansion: Cross-Linguality](#attributions-cross-linguality)
- [**SpaceShaping**](#space-shaping)
    - [Idea](#space-shaping-idea)
    - [Example](#space-shaping-example)
- [**Symbolic**](#symbolic)
    - [AMR parsing and multi-subgraph metric](#amr)
- [FAQ](#faq)
- [Citation](#citation)

## Installation<a id="requirements"></a>

You can install via pip with:
```

pip install xplainsim
```

## Attributions <a id="attribution"></a>

### Example Code<a id="attributions-example"></a>

```python
from xplain.attribution import ModelFactory
print(ModelFactory.show_options()) # shows available model names, use in build below
model = ModelFactory.build("huggingface_id") # e.g sentence-transformers/all-mpnet-base-v2
texta = 'The dog runs after the kitten in the yard.'
textb = 'Outside in the garden the cat is chased by the dog.'
A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```

### Expansion: Subtokens-to-Tokens<a id="attributions-subtokens-to-tokens"></a>
```python
# same as above, then
A, tokens_a, tokens_b = model.postprocess_attributions(A, tokens_a, tokens_b, sparsification_method="FlowAlign")
```

### Expansion: Cross-Linguality <a id="attributions-cross-linguality"></a>
```python
from xplain.attribution import ModelFactory
print(ModelFactory.show_options()) # shows available model names, use in build below
model = ModelFactory.build("huggingface_id") # a multilingual model, e.g Alibaba-NLP/gte-multilingual-base
texta = 'The dog runs after the kitten in the yard.'
textb = 'Im Garten rennt der Hund der Katze hinterher.'
A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```

## SpaceShaping<a id="space-shaping"></a>

### Idea<a id="space-shaping-idea"></a>

The idea is as follows: We start with

- `my_metrics`: A bunch of interpreatble measures that should be reflected in the embedding space.
- `documents1`, `documents2`: Two lists with documents in string format.
- `feature_names`, `feature_dims`: The name for each metric/feature and the number of dimensions it should be assigned.

The following is Pseudo code, for an actual demo see further [below](space-shaping-toy).

```python
from sentence_transformers import InputExample
from xplain.spaceshaping import PartitionedSentenceTransformer

examples = []

# compute the training/partitioning target
for x, y in zip(list_with_strings, other_list_with_strings):
	similarities = []
	for metric in my_metrics:
		similarities.append(metric.score(x, y))
	examples.append(InputExample(texts=[x, y], label=similarities))

# instantiate model and train
pt = PartitionedSentenceTransformer(feature_names, feature_dims)

pt.train_model(examples)
```


### Space Paritioning Example<a id="space-shaping-example"></a>

Here's a very simple example for training and inferring with a custom model.

Concretely, we paritition the embedding into three features/parts

1. Bag-of-words: Learns to reflect bag-of-words distance
2. Named entity similarity: Learns to reflect similarity of named entities
3. (Not explicitly trained): Residual features for capturing the semantic similarity that makes for "the rest"

Note that this is only a toy code, and the training happens on little data, however, the feature paritioning will already have some effect.

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

# let's build our target metrics that should be reflected within the embedding space,
def bow_sim(x1, x2):
	x1 = set(x1.split())
	x2 = set(x2.split())
	inter = x1.intersection(x2)
	union = x1.union(x2)
	return len(inter) / len(union)

def ner_sim(doc1, doc2):
	x1_ner = " ".join([ne.text for ne in doc1.ents])
	x2_ner = " ".join([ne.text for ne in doc2.ents])
	if not x1_ner and not x2_ner:
		return 1.0
	return bow_sim(x1_ner, x2_ner)

docs1, docs2 = [nlp(x) for x, _ in some_pairs], [nlp(y) for _, y in some_pairs]
target = [[bow_sim(x1, x2), ner_sim(docs1[i], docs2[i])] for i, (x1, x2) in enumerate(some_pairs)]
some_examples = [InputExample(texts=[x1, x2], label=target[i]) for (i, (x1, x2)) in enumerate(some_pairs)]

docs1_dev, docs2_dev = [nlp(x) for x, _ in some_pairs_dev], [nlp(y) for _, y in some_pairs_dev]
target_dev = [[bow_sim(x1, x2), ner_sim(docs1_dev[i], docs2_dev[i])] for i, (x1, x2) in enumerate(some_pairs_dev)]
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
pt.train_model(some_examples, some_examples_dev)

# eval correlation to custom metric after train
json = pt.explain_similarity([x for x, y in some_pairs_dev], [y for x, y in some_pairs_dev])
print(pearsonr([x.label[0] for x in some_examples_dev], [dic["bow"] for dic in json]))
print(pearsonr([x.label[1] for x in some_examples_dev], [dic["ner"] for dic in json]))

# print a toy example after training
print(pt.explain_similarity(["The kitten drinks milk"], ["A cat slurps something"]))
```

## Symbolic<a id="symbolic"></a>

### AMR Parsing and Multi-Subgraph Metric<a id="amr"></a>

The approch consists roughly in two steps:

1. Parse each input text to an Abstract Meaning Representation Graph
2. Match those Meaning Graphs with Graph Similarity Metrics, also with regard to aspectual subgraphs as elicited in AMR (e.g., Agent, Patient, Negation,...)

```python
from xplain.symbolic.model import AMRSimilarity
explainer = AMRSimilarity()
sents1 = ["Barack Obama holds a talk"]
sents2 = ["Hillary Clinton holds a talk"]
exp = explainer.explain_similarity(sents1, sents2)
print(exp)
```

This will print a json dictionary with aspectual graph matching scores. 
To also return the graphs and aspectual subgraphs, use `return_graphs=True` in `explain_similarity`.

## FAQ<a id="faq"></a>

## Citation<a id="citation"></a>
