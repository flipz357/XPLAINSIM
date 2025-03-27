import logging

logger = logging.getLogger(__name__)

class AMRSimilarity():

    def __init__(self, parser_engine=None, measure=None, subgraph_extractor=None):
        
        if parser_engine is None:
            parser, reader, standardizer = self._build_parser_engine()
        
        if measure is None:
            measure = self._build_measure()
        
        if subgraph_extractor is None:
            subgraph_extractor = self._build_subgraph_extractor()

        self.parser = parser
        self.reader = reader
        self.standardizer = standardizer
        self.measure = measure
        self.subgraph_extractor = subgraph_extractor

    
    @staticmethod
    def _build_measure():
        from smatchpp import Smatchpp
        class DummyReader():
            def string2graph(self, input):
                return input

        dummy_reader = DummyReader()

        measure = Smatchpp(graph_reader=dummy_reader)
        return measure
    
    @staticmethod
    def _build_subgraph_extractor():
        from smatchpp.formalism.amr import tools as amrtools
        subgraph_extractor = amrtools.AMRSubgraphExtractor()
        return subgraph_extractor
    
    @staticmethod
    def _build_parser_engine():
        import amrlib
        try:
            stog = amrlib.load_stog_model()
        except FileNotFoundError:
            url = "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz"
            logger.warning("Parser Model not accessible. I will try to download and install a default model from {}.".format(url))
            amrlibpath = amrlib.__file__
            amrlibpath = "/".join(amrlibpath.split("/")[:-1]) +"/"
            import subprocess
            subprocess.call(["mkdir", amrlibpath + "data"])
            import urllib.request
            urllib.request.urlretrieve(url, amrlibpath + "data/" + "model_parse_xfm_bart_base-v0_1_0.tar.gz")
            subprocess.call(["tar", "-xvzf", amrlibpath + "data/model_parse_xfm_bart_base-v0_1_0.tar.gz", "-C", amrlibpath + "data/"])
            subprocess.call(["mv", amrlibpath + "data/model_parse_xfm_bart_base-v0_1_0", amrlibpath + "data/model_stog"])
            stog = amrlib.load_stog_model()
        parser = stog
        
        from smatchpp import data_helpers
        reader = data_helpers.PenmanReader()
        from smatchpp.formalism.amr import tools as amrtools
        standardizer = amrtools.AMRStandardizer()        
        return parser, reader, standardizer
    
    def _raw_string_graph_to_subgraph_dict(self, string_graph_raw):
        string_graph = "\n".join([x for x in string_graph_raw.split("\n") if not x.startswith("#")])
        g = self.reader.string2graph(string_graph)
        g = self.standardizer.standardize(g)
        name_subgraph_dict = self.subgraph_extractor.all_subgraphs_by_name(g)
        name_subgraph_dict["global"] = self.standardizer.standardize(self.reader.string2graph(string_graph))
        return name_subgraph_dict

    def explain_similarity(self, xsent: list, ysent:list, return_graphs=None):
        graphs1 = self.parser.parse_sents(xsent)
        graphs2 = self.parser.parse_sents(ysent)
        explanations = []
        for string_graph1_raw, string_graph2_raw in zip(graphs1, graphs2):
            name_subgraph_dict1 = self._raw_string_graph_to_subgraph_dict(string_graph1_raw)
            name_subgraph_dict2 = self._raw_string_graph_to_subgraph_dict(string_graph2_raw)
            result = {}
            for graph_type in name_subgraph_dict1:
                g1s = name_subgraph_dict1[graph_type]
                g2s = name_subgraph_dict2[graph_type]
                result[graph_type] = self.measure.score_pair(g1s, g2s)
                if return_graphs:
                    result[graph_type]["subgraph1"] = g1s
                    result[graph_type]["subgraph2"] = g2s
            explanations.append(result)
        return explanations
