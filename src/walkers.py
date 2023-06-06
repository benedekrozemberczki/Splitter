"""DeepWalker class."""

import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from gensim.models import Word2Vec

class DeepWalker(object):
    """
    DeepWalk node embedding learner object.
    A barebones implementation of "DeepWalk: Online Learning of Social Representations".
    Paper: https://arxiv.org/abs/1403.6652
    Video: https://www.youtube.com/watch?v=aZNtHJwfIVg
    """
    def __init__(self, graph, args):
        """
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        while len(walk) < self.args.walk_length:
            nebs = [n for n in nx.neighbors(self.graph, walk[-1])]
            if len(nebs) == 0:
                break
            walk.append(random.choice(nebs))
        return walk

    def create_features(self):
        """
        Creating random walks from each node.
        """
        self.paths = []
        for node in tqdm(self.graph.nodes()):
            for _ in range(self.args.number_of_walks):
                walk = self.do_walk(node)
                self.paths.append(walk)

    def learn_base_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.paths = [[str(node) for node in walk] for walk in self.paths]

        model = Word2Vec(self.paths,
                         vector_size=self.args.dimensions,
                         window=self.args.window_size,
                         min_count=1,
                         sg=1,
                         workers=self.args.workers,
                         iter=1)

        self.embedding = np.array([list(model[str(n)]) for n in self.graph.nodes()])
        return self.embedding
