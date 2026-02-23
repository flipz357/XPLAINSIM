import logging
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)

class AMRSimilarity:

    def __init__(self, parser_engine=None, measure=None, subgraph_extractor=None, graph_scorer=None):
        
        if parser_engine is None:
            parser, reader, standardizer = self._build_parser_engine()
        else:
            parser, reader, standardizer = parser_engine

        if measure is None:
            measure = self._build_measure()
        
        #if graph_scorer is None:
        #    measure = self._build_graph_scorer()
        
        if subgraph_extractor is None:
            subgraph_extractor = self._build_subgraph_extractor()

        self.parser = parser
        self.reader = reader
        self.standardizer = standardizer
        self.measure = measure
        self.subgraph_extractor = subgraph_extractor
        self.graph_scorer = graph_scorer

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
    def _build_graph_scorer():
        try:
            from smatchpp import score
        except ModuleNotFoundError as e:
            from xplain.symbolic.parser_install import _AMRLIB_SMATCHPP_INSTALL_MESSAGE
            raise ModuleNotFoundError(_AMRLIB_SMATCHPP_INSTALL_MESSAGE) from e
        
        scorer = score.TripleScorer()
        return scorer
    
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

    def explain_similarity(self, 
                           sent_a: Union[str, List[str]], sent_b: Union[str, List[str]], 
                           return_graphs: bool = False
                           ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        
        single_input = False

        if isinstance(sent_a, str) and isinstance(sent_b, str):
            sent_a = [sent_a]
            sent_b = [sent_b]
            single_input = True
        elif isinstance(sent_a, str) or isinstance(sent_b, str):
            raise ValueError("sent_a and sent_b must both be strings or both lists")

        if len(sent_a) != len(sent_b):
            raise ValueError("sent_a and sent_b must have the same length")
        
        graphs1 = self.parser.parse_sents(sent_a)
        graphs2 = self.parser.parse_sents(sent_b)
        explanations = []
        i = 0
        for sa, sb, string_graph1_raw, string_graph2_raw in zip(sent_a, sent_b, graphs1, graphs2):
            name_subgraph_dict1 = self._raw_string_graph_to_subgraph_dict(string_graph1_raw)
            name_subgraph_dict2 = self._raw_string_graph_to_subgraph_dict(string_graph2_raw)
            result = {"sent_a": sa, "sent_b": sb}
            for graph_type in name_subgraph_dict1:
                g1s = name_subgraph_dict1[graph_type]
                g2s = name_subgraph_dict2[graph_type]
                
                g1 = self.measure.graph_reader.string2graph(g1s)
                g1 = self.measure.graph_standardizer.standardize(g1)
                g2 = self.measure.graph_reader.string2graph(g2s)
                g2 = self.measure.graph_standardizer.standardize(g2)
                g1, g2, v1, v2 = self.measure.graph_pair_preparer.prepare_get_vars(g1, g2)
                alignment, var_index, _ = self.measure.graph_aligner.align(g1, g2, v1, v2)
                var_map = self.measure.graph_aligner._get_var_map(alignment, var_index)
                interpretable_mapping = self.measure.graph_aligner._interpretable_mapping(var_map, g1, g2)
                # array with 4 values: match count from left to right, from right to left, size of left, size of right 
                # first two values typically the same
                score_stats = self.measure.graph_scorer.score(g1, g2, alignment, var_index)
                # a bit ad hoc but perhaps less confusing than recall, precision and F1
                similarity = (score_stats[0] + score_stats[1]) / max(score_stats[2] + score_stats[3], 1)
                match = {"similarity": similarity}
                result[graph_type] = match
                """
                result[graph_type] = self.measure.score_pair(g1s, g2s)
                """
                if return_graphs:
                    result[graph_type]["subgraph1"] = g1s
                    result[graph_type]["subgraph2"] = g2s
                    result[graph_type]["alignment"] = interpretable_mapping
            explanations.append(result)
            i += 1
        return explanations[0] if single_input else explanations
