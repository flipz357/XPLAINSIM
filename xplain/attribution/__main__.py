import argparse
from xplain.attribution import utils, ReferenceTransformer

def model_factory(model_uri, reference_uri):
    model = None
    if model_uri == "XSMPNet":
        from xplain.attribution import XSMPNet
        model = XSMPNet

    return model, ReferenceTransformer(reference_uri)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog="GenerateAttributions",
                    description="Generate Approximate Attributions for an input text pair")
    parser.add_argument("-texta", type=str, default='The dog runs after the kitten in the yard.')
    parser.add_argument("-textb", type=str, default='Outside in the garden the cat is chased by the dog.')
    parser.add_argument("-model", default="XSMPNet")
    parser.add_argument("-reference_transformer", default="sentence-transformers/all-mpnet-base-v2")
    
    args = parser.parse_args()
    from sentence_transformers.models import Pooling
    
    model, transformer = model_factory(args.model, args.reference_transformer)
    pooling = Pooling(transformer.get_word_embedding_dimension())
    model = model(modules=[transformer, pooling])
    #model.to(torch.device('cuda:1'))
    model.reset_attribution()
    model.init_attribution_to_layer(idx=10, N_steps=50)
    A, tokens_a, tokens_b = model.explain_similarity(args.texta, args.textb, move_to_cpu=True, sim_measure='cos')
    

