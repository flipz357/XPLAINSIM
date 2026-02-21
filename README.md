 # XPLAINSIM: A Toolkit for Explaining Text Similarity

A research toolkit for decomposing and **explaining text similarity**
across neural, structured, and symbolic levels. 

It provides:
- Token-level attribution for neural similarity models
- Feature-partitioned neural embedding spaces (Space Shaping)
- Graph-based symbolic similarity via Abstract Meaning Representation

The toolkit is designed for interpretability research,
controlled embedding and metric alignment, and hybrid neural-symbolic text analysis.

The toolkit is also modular: each explanation paradigm can be used independently or combined in hybrid setups.


## Conceptual Overview

XPLAINSIM provides three complementary explanation paradigms:

| Module        | Explanation Level  | What it Does     |
|---------------|--------------------|------------------|
| Attribution   | Token level        | Explain which tokens drive similarity |
| SpaceShaping  | Embedding space    | Shape dimensions to encode custom aspects |
| Symbolic      | Graph level        | Explain which semantic roles/aspects align |

## Overview of Repository / Table of Contents

- [Installation](#requirements)
- [**Attributions**](#attribution)
    - [Idea](#attribution-idea)
    - [Examples](#attributions-example)
- [**Space Shaping**](#space-shaping)
    - [Idea](#space-shaping-idea)
    - [Examples](#space-shaping-example)
- [**Symbolic**](#symbolic)
    - [Idea](#symbolic-idea)
    - [Example](#symbolic-example)
- [FAQ](#faq)
- [Citation](#citation)

## Installation<a id="requirements"></a>

You can install via pip with:

```
pip install xplainsim
```

That's it. Only when using the Symbolic module with the default parser one [small extra installation](#symbolic-extra-install) is necessary.

## Attributions <a id="attribution"></a>

### Idea<a id="attribution-idea"></a>

Token-level attribution decomposes embedding similarity into fine-grained token interactions between two texts.

Given a neural embedding model and two texts we trace the similarity back to interactions of individual input tokens. 

The explanation is a matrix over the tokens from each input (the sum of this matrix approximates the similarity of the embeddings).

### Example<a id="attributions-example"></a>

#### Show Currently Available Models

```python
print(ModelFactory.show_options()) # shows available model names, use in build below
```

#### Compute Attributions

```python
from xplain.attribution import ModelFactory
model = ModelFactory.build("sentence-transformers/all-mpnet-base-v2") # use print(ModelFactory.show_options()) to show others
texta = 'The dog runs after the kitten in the yard.'
textb = 'Outside in the garden the cat is chased by the dog.'
A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```

Example output structure:

- `A`: token-level contribution matrix
- `tokens_a`: token list for text A
- `tokens_b`: token list for text B

#### Expansion: Subtokens-to-Tokens<a id="attributions-subtokens-to-tokens"></a>
```python
# same as above, then
A, tokens_a, tokens_b = model.postprocess_attributions(A, tokens_a, tokens_b, sparsification_method="FlowAlign")
```

#### Expansion: Cross-Linguality <a id="attributions-cross-linguality"></a>
```python
from xplain.attribution import ModelFactory
model = ModelFactory.build("Alibaba-NLP/gte-multilingual-base") # use print(ModelFactory.show_options()) to show others
texta = 'The dog runs after the kitten in the yard.'
textb = 'Im Garten rennt der Hund der Katze hinterher.'
A, tokens_a, tokens_b = model.explain_similarity(texta, textb, move_to_cpu=True, sim_measure='cos')
```


## Space Shaping<a id="space-shaping"></a>

### Idea<a id="space-shaping-idea"></a>

Space Shaping enforces interpretable structure inside embedding spaces.

Instead of learning a monolithic embedding, the vector is partitioned into
dedicated subspaces, each trained to reflect a predefined interpretable metric
(e.g., bag-of-words overlap, named entity similarity, sentiment, etc.).

This enables:
- Controllable similarity decomposition
- Feature-aligned embeddings
- Hybrid symbolic–neural objectives

```python
from sentence_transformers import InputExample
from xplain.spaceshaping import PartitionedSentenceTransformer

examples = []

# compute the training/partitioning target
for x, y in zip(list_with_strings, other_list_with_strings):
	similarities = []
        # Metrics/aspects that should be reflected in the embedding space
	for metric in my_metrics:
		similarities.append(metric.score(x, y))
	examples.append(InputExample(texts=[x, y], label=similarities))

# instantiate model and train
pt = PartitionedSentenceTransformer(feature_names, feature_dims)

pt.train_model(examples)
```

### Space Partitioning Example<a id="space-shaping-example"></a>

Here's a very simple example for training and inferring with a custom model.

Concretely, we partition the embedding into three features/parts

1. Bag-of-words: Learns to reflect bag-of-words distance
2. Named entity similarity: Learns to reflect similarity of named entities
3. (Not explicitly trained): Residual features for capturing the semantic similarity that makes for "the rest"

Note that this is only a toy code, and the training happens on little data, however, the feature partitioning will already have some effect.

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

# explanation can be called before training, but it's meaningless, just to compare to later
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


### Idea<a id="symbolic-idea"></a>

Unlike pure neural similarity, this approach decomposes similarity
along semantic roles (Agent, Patient, Negation, etc.), enabling
aspect-level semantic comparison.

This is based on comparing AMR graphs of texts. Abstract Meaning Representation (AMR) encodes sentence meaning as a graph of concepts and semantic roles.

**Installation note**<a id="symbolic-extra-install"></a>: 

For using the Symbolic module with the default parser one small extra installation is necessary:

```
xplain-install-amr
```

Ensure also that for this `transformers<5` is installed, as the default AMR parser is not yet compatible with version 5.

### Example<a id="amr"></a>

The approach consists roughly in two steps:

1. Parse each input text to an AMR Graph that expresses the text semantics in a symbolic way
2. Match those Meaning Graphs with Graph Similarity Metrics to elicit meaning similarity aspects (e.g., Agent, Patient, Negation,...)

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
