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
        stog = amrlib.load_stog_model()
        parser = stog
        
        from smatchpp import data_helpers
        reader = data_helpers.PenmanReader()
        
        from smatchpp.formalism.amr import tools as amrtools
        standardizer = amrtools.AMRStandardizer()
        
        return parser, reader, standardizer

    def explain_similarity(self, xsent: list, ysent:list, return_graphs=None):
        graphs1 = self.parser.parse_sents(xsent)
        graphs2 = self.parser.parse_sents(ysent)
        print(graphs1[0])
        print(graphs2[0])
        explanations = []
        for string_graph1_raw, string_graph2_raw in zip(graphs1, graphs2):

            string_graph1 = "\n".join([x for x in string_graph1_raw.split("\n") if not x.startswith("#")])
            g1 = self.reader.string2graph(string_graph1)
            g1 = self.standardizer.standardize(g1)
            name_subgraph_dict1 = self.subgraph_extractor.all_subgraphs_by_name(g1)
            name_subgraph_dict1["full"] = self.standardizer.standardize(self.reader.string2graph(string_graph1))

            string_graph2 = "\n".join([x for x in string_graph2_raw.split("\n") if not x.startswith("#")])
            g2 = self.reader.string2graph(string_graph2)
            g2 = self.standardizer.standardize(g2)
            name_subgraph_dict2 = self.subgraph_extractor.all_subgraphs_by_name(g2)
            name_subgraph_dict2["full"] = self.standardizer.standardize(self.reader.string2graph(string_graph2))

            result = {}

            for graph_type in name_subgraph_dict1:
                g1s = name_subgraph_dict1[graph_type]
                g2s = name_subgraph_dict2[graph_type]
                result[graph_type] = self.measure.score_pair(g1s, g2s)
                if return_graphs:
                    result[graph_type]["subgraph1"] = g1s
                    result[graph_type]["subgraph2"] = g2s
            explanations.append(result)

        return result
