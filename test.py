def test_space_shaping():
    from scipy.stats import pearsonr
    from xplain.spaceshaping import PartitionedSentenceTransformer
    from sentence_transformers import InputExample

    pt = PartitionedSentenceTransformer(feature_names=["bow"], feature_dims=[100])
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts")
    some_pairs = list(
        zip(
            [dic["sentence1"] for dic in ds["train"]],
            [dic["sentence2"] for dic in ds["train"]],
        )
    )

    def bow_sim(x1, x2):
        x1 = set(x1.split())
        x2 = set(x2.split())
        inter = x1.intersection(x2)
        union = x1.union(x2)
        return len(inter) / len(union)

    target = [[bow_sim(x1, x2)] for x1, x2 in some_pairs]
    some_examples = [
        InputExample(texts=[x1, x2], label=target[i])
        for (i, (x1, x2)) in enumerate(some_pairs)
    ]
    json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])
    print(json)
    print(pearsonr([x.label[0] for x in some_examples], [dic["bow"] for dic in json]))
    pt.train(some_examples, some_examples)
    json = pt.explain_similarity([x for x, y in some_pairs], [y for x, y in some_pairs])
    print(json)
    print(pearsonr([x.label[0] for x in some_examples], [dic["bow"] for dic in json]))


def test_attribution():
    from xplain.attribution import ModelFactory
    from xplain.attribution import plot_attributions
    import torch
    
    print(ModelFactory.show_options()) # shows available model names, use in build below
    model = ModelFactory.build("gte-multilingual-base")
    texta = "The dog runs after the kitten in the yard."
    textb = "Im Garten rennt der Hund der Katze hinterher."
    andrianos_test = True
    device = torch.device("mps") if andrianos_test else torch.device("cuda:0")
    A, tokens_a, tokens_b = model.explain_similarity(
        texta, textb, move_to_cpu=True, sim_measure="cos", device=device, postprocess_trim_if_no_postprocessing=True
    )
    # tests for the postprocessing.
    f = plot_attributions(
        A,
        tokens_a,
        tokens_b,
        size=(5, 5),
        # range=.3,
        show_colorbar=True,
        shrink_colorbar=0.5,
    )
    f.savefig("NoPostprocess.png", dpi=300, bbox_inches="tight")
    # A, tokens_a, tokens_b = model.explain_similarity(
    #     texta,
    #     textb,
    #     move_to_cpu=True,
    #     sim_measure="cos",
    #     postprocess_sparsify="SimpleAlign",
    #     device=torch.device("mps"),
    # )
    # f = plot_attributions(
    #     A,
    #     tokens_a,
    #     tokens_b,
    #     size=(5, 5),
    #     # range=.3,
    #     show_colorbar=True,
    #     shrink_colorbar=0.5,
    # )
    # f.savefig("MaxAlignplot.png", dpi=300, bbox_inches="tight")
    A, tokens_a, tokens_b = model.explain_similarity(
        texta,
        textb,
        move_to_cpu=True,
        sim_measure="cos",
        postprocess_sparsify="FlowAlign",
        device=torch.device("mps"),
    )
    f = plot_attributions(
        A,
        tokens_a,
        tokens_b,
        size=(5, 5),
        # range=.3,
        show_colorbar=True,
        shrink_colorbar=0.5,
    )
    f.savefig("FlowAlign.png", dpi=300, bbox_inches="tight")
    
def test_all_attribution_models_compile():
    from xplain.attribution import ModelFactory
    from xplain.attribution import plot_attributions
    import torch
    
    texta = "The dog runs after the kitten in the yard."
    textb = "Outside in the garden the cat is chased by the dog."
    andrianos_test = True
    neg_index = True
    print("All available models are " + str(ModelFactory.show_options()))
    device = torch.device("mps") if andrianos_test else torch.device("cuda:0")
    for working_model_name in ModelFactory.show_options():
        model = ModelFactory.build(working_model_name, idx=-2 if neg_index==True else 10)
        A, tokens_a, tokens_b = model.explain_similarity(
        texta, textb, move_to_cpu=True, sim_measure="cos", device=device, postprocess_trim_if_no_postprocessing=True
        )
        print("Succesfully loaded and predicted an attribution with model: " + working_model_name)
    
    print("All models were compiled succesfully.")


def test_symbolic():
    """
    import amrlib
    stog = amrlib.load_stog_model()
    sents1 = ["The cat does not drink milk."]
    sents2 = ["The cat drinks milk."]
    sents1 = ["Barack Obama holds a talk"]
    sents2 = ["Hillary Clinton holds a talk"]
    graphs1 = stog.parse_sents(sents1)
    graphs2 = stog.parse_sents(sents2)
    from smatchpp import Smatchpp, data_helpers
    from smatchpp.formalism.amr import tools as amrtools
    reader = data_helpers.PenmanReader()

    class DummyReader():
        def string2graph(self, input):
            return input

    dummy_reader = DummyReader()

    standardizer = amrtools.AMRStandardizer()
    subgraph_extractor = amrtools.AMRSubgraphExtractor()
    measure = Smatchpp(graph_reader=dummy_reader)

    for string_graph1_raw, string_graph2_raw in zip(graphs1, graphs2):

        string_graph1 = "\n".join([x for x in string_graph1_raw.split("\n") if not x.startswith("#")])
        g1 = reader.string2graph(string_graph1)
        g1 = standardizer.standardize(g1)
        name_subgraph_dict1 = subgraph_extractor.all_subgraphs_by_name(g1)
        name_subgraph_dict1["full"] = standardizer.standardize(reader.string2graph(string_graph1))

        string_graph2 = "\n".join([x for x in string_graph2_raw.split("\n") if not x.startswith("#")])
        g2 = reader.string2graph(string_graph2)
        g2 = standardizer.standardize(g2)
        name_subgraph_dict2 = subgraph_extractor.all_subgraphs_by_name(g2)
        name_subgraph_dict2["full"] = standardizer.standardize(reader.string2graph(string_graph2))

        result = {}

        for graph_type in name_subgraph_dict1:
            g1s = name_subgraph_dict1[graph_type]
            g2s = name_subgraph_dict2[graph_type]
            result[graph_type] = measure.score_pair(g1s, g2s)
        print(string_graph1_raw)
        print(string_graph2_raw)
        print("main", result["full"]["main"]["F1"])
        print("negation", result["POLARITY"]["main"]["F1"] )
        print("focus", result["FOCUS"]["main"]["F1"])
        print("NER", result["NER"]["main"]["F1"])
    """
    from xplain.symbolic.model import AMRSimilarity

    explainer = AMRSimilarity()
    sents1 = ["Barack Obama holds a talk"]
    sents2 = ["Hillary Clinton holds a talk"]
    exp = explainer.explain_similarity(sents1, sents2, return_graphs=True)
    print(exp)

test_all_attribution_models_compile()
# test_attribution()
# test_space_shaping()
# test_symbolic()
