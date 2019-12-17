"""Running the Splitter."""

import torch
from param_parser import parameter_parser
from splitter import SplitterTrainer
from utils import tab_printer, graph_reader

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    trainer = SplitterTrainer(graph, args)
    trainer.fit()
    trainer.save_embedding()
    trainer.save_persona_graph_mapping()

if __name__ == "__main__":
    main()
