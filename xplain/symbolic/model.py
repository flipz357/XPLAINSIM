import logging

logger = logging.getLogger(__name__)

class AMRSimilarity:

    def __init__(self, parser_engine=None, measure=None, subgraph_extractor=None):
        
        if parser_engine is None:
            parser, reader, standardizer = self._build_parser_engine()
        else:
            parser, reader, standardizer = parser_engine

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
        try:
            from smatchpp import Smatchpp
        except ModuleNotFoundError as e:
            from xplain.symbolic.parser_install import _AMRLIB_SMATCHPP_INSTALL_MESSAGE
            raise ModuleNotFoundError(_AMRLIB_SMATCHPP_INSTALL_MESSAGE) from e
        
        class DummyReader():
            def string2graph(self, input):
                return input

        dummy_reader = DummyReader()

        measure = Smatchpp(graph_reader=dummy_reader)
        return measure
    
    @staticmethod
    def _build_subgraph_extractor():
        try:
            from smatchpp.formalism.amr import tools as amrtools
        except ModuleNotFoundError as e:
            from xplain.symbolic.parser_install import _AMRLIB_SMATCHPP_INSTALL_MESSAGE
            raise ModuleNotFoundError(_AMRLIB_SMATCHPP_INSTALL_MESSAGE) from e
        subgraph_extractor = amrtools.AMRSubgraphExtractor()
        return subgraph_extractor
    
    @staticmethod
    def _build_parser_engine():
        try:
            import amrlib
        except ModuleNotFoundError as e:
            from xplain.symbolic.parser_install import _AMRLIB_SMATCHPP_INSTALL_MESSAGE
            raise ModuleNotFoundError(_AMRLIB_SMATCHPP_INSTALL_MESSAGE) from e
        try:
            stog = amrlib.load_stog_model()
        except FileNotFoundError:
                raise RuntimeError(
        "AMR parser model not found.\n"
        "Run `xplain-install-amr` to install the default model.")
        except AttributeError:
            raise AttributeError ( "AMRlib might not work with newer versions of transformers, please install, e.g., transformers==4.49.0")

        parser = stog
        
        try:
            from smatchpp import data_helpers
        except ModuleNotFoundError as e:
            from xplain.symbolic.parser_install import _AMRLIB_SMATCHPP_INSTALL_MESSAGE
            raise ModuleNotFoundError(_AMRLIB_SMATCHPP_INSTALL_MESSAGE) from e
        
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

    def explain_similarity(self, xsent: list[str], ysent:list[str], return_graphs: bool = False) -> list[dict]:
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
