import json
import torch 
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from  walkers import DeepWalker
from ego_splitting import EgoNetSplitter

class Splitter(torch.nn.Module):
     """
     An implementation of "Splitter: Learning Node Representations that Capture Multiple Social Contexts" (WWW 2019).
     Paper: http://epasto.org/papers/www2019splitter.pdf
     """
     def __init__(self, args, base_node_count, node_count):
         """
         Splliter set up.
         :param args: Arguments object.
         :param base_node_count: Number of nodes in the source graph.
         :param node_count: Number of nodes in the persona graph.
         """
         super(Splitter, self).__init__()
         self.args = args
         self.base_node_count = base_node_count
         self.node_count = node_count

     def create_weights(self):
         """
         Creating weights for embedding.
         """
         self.base_node_embedding = torch.nn.Embedding(self.base_node_count, self.args.dimensions, padding_idx = 0)
         self.node_embedding = torch.nn.Embedding(self.node_count, self.args.dimensions, padding_idx = 0)
         self.node_noise_embedding = torch.nn.Embedding(self.node_count, self.args.dimensions, padding_idx = 0)

     def initialize_weights(self, base_node_embedding, mapping):
         """
         Using the base embedding and the persona mapping for initializing the embedding matrices.
         :param base_node_embedding: Node embedding of the source graph.
         :param mapping: Mapping of personas to nodes.
         """
         persona_embedding = np.array([base_node_embedding[original_node] for node, original_node in mapping.items()])
         self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
         self.node_noise_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
         self.base_node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(base_node_embedding),requires_grad=False)

     def calculate_main_loss(self, sources, contexts, targets):
         """
         Calculating the main embedding loss.
         :param sources: Source node vector.
         :param contexts: Context node vector.
         :param targets: Binary target vector.
         :return main_loss: Loss value.
         """
         node_f = self.node_embedding(sources)
         node_f = torch.t(torch.t(node_f) /torch.norm(node_f , p=2, dim=1))
         feature_f = self.node_noise_embedding(contexts)
         feature_f = torch.t(torch.t(feature_f) /torch.norm(feature_f , p=2, dim=1))
         scores = torch.sum(node_f * feature_f,dim=1) 
         scores = torch.exp(scores)/(1+torch.exp(scores))
         main_loss = targets*torch.log(scores) + (1-targets)*torch.log(1-scores)
         main_loss = -torch.mean(main_loss)
         return main_loss

     def calculate_regularization(self, pure_sources, personas):
         """
         Calculating the regularization loss.
         :param pure_sources: Source nodes in persona graph.
         :param personas: Context node vector.
         :return regularization_loss: Loss value.
         """
         source_f = self.node_embedding(pure_sources)
         original_f = self.base_node_embedding(personas)
         source_f = source_f/torch.norm(source_f , p=2, dim=0)
         original_f = original_f /torch.norm(original_f , p=2, dim=0)
         scores = torch.sum(source_f  * original_f,dim=1)
         scores = torch.exp(scores)/(1+torch.exp(scores))
         regularization_loss = -torch.mean(torch.log(scores))
         return regularization_loss

     def forward(self, sources, contexts, targets, personas, pure_sources):
         main_loss = self.calculate_main_loss(sources, contexts, targets)
         regularization_loss = self.calculate_regularization(pure_sources, personas)
         loss = main_loss + self.args.lambd*regularization_loss
         return loss
         
class SplitterTrainer(object):

    def __init__(self, graph, args):
        self.graph = graph
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def base_model_fit(self):
        self.base_walker = DeepWalker(self.graph, self.args)
        print("\nDoing base random walks.\n")
        self.base_walker.create_features()
        print("\nLearning the base model.\n")
        self.base_node_embedding = self.base_walker.learn_base_embedding()
        print("\nDeleting the base walker.\n")
        del self.base_walker

    def create_split(self):
        self.egonet_splitter = EgoNetSplitter(self.graph)
        self.persona_walker = DeepWalker(self.egonet_splitter.persona_graph, self.args)
        print("\nDoing persona random walks.\n")
        self.persona_walker.create_features()

    def setup_model(self):
        base_node_count = self.graph.number_of_nodes()
        persona_node_count = self.egonet_splitter.persona_graph.number_of_nodes()
        self.model = Splitter(self.args, base_node_count, persona_node_count)
        self.model.create_weights()
        self.model.initialize_weights(self.base_node_embedding, self.egonet_splitter.personality_map)
        self.model = self.model.to(self.device)

    def reset_node_sets(self):
        self.pure_sources = []
        self.personas = []
        self.sources = []
        self.contexts = []
        self.targets = [] 

    def create_batch(self, source_node, context_node):
        self.pure_sources = self.pure_sources + [source_node]
        self.personas = self.personas + [self.egonet_splitter.personality_map[source_node]]
        self.sources  = self.sources + [source_node]+[source_node]*self.args.negative_samples
        self.contexts = self.contexts + [context_node] + random.sample(self.graph.nodes(),self.args.negative_samples)
        self.targets = self.targets + [1.0] + [0.0]*self.args.negative_samples

    def transfer_batch(self):
        self.sources = torch.LongTensor(self.sources).to(self.device)
        self.contexts = torch.LongTensor(self.contexts).to(self.device)
        self.targets = torch.FloatTensor(self.targets).to(self.device)
        self.personas = torch.LongTensor(self.personas).to(self.device)
        self.pure_sources = torch.LongTensor(self.pure_sources).to(self.device)

    def optimize(self):
        loss = self.model(self.sources, self.contexts, self.targets, self.personas, self.pure_sources)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.reset_node_sets()
        return loss.item()

    def fit(self):
        self.reset_node_sets()
        self.base_model_fit()
        self.create_split()
        self.setup_model()
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        print("\nLearning the joint model.\n")
        random.shuffle(self.persona_walker.paths)
        self.steps = 0
        self.losses = 0
        self.walk_steps = trange(len(self.persona_walker.paths), desc="Loss")
        for step in self.walk_steps:
            if step % 1000 ==0:
                self.steps = 0
                self.losses = 0
            walk = self.persona_walker.paths[step]
            for i in range(self.args.walk_length-self.args.window_size):
                for j in range(1,self.args.window_size+1):
                    source_node = walk[i]
                    context_node = walk[i+j]
                    self.create_batch(source_node, context_node)
            for i in range(self.args.window_size,self.args.walk_length):
                for j in range(1,self.args.window_size+1):
                    source_node = walk[i]
                    context_node = walk[i-j]
                    self.create_batch(source_node, context_node)
            self.transfer_batch()
            self.losses = self.losses + self.optimize()
            self.steps = self.steps + 1
            average_loss = self.losses/self.steps
            self.walk_steps.set_description("Splitter (Loss=%g)" % round(average_loss,4))

    def save_embedding(self):
        """
        Saving the node embedding.
        """
        print("\n\nSaving the model.\n")
        nodes = torch.LongTensor([node for node in self.egonet_splitter.persona_graph.nodes()]).to(self.device)
        self.embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        self.embedding  = np.concatenate([np.array(range(self.embedding.shape[0])).reshape(-1,1),self.embedding],axis=1)
        self.embedding = pd.DataFrame(self.embedding, columns = embedding_header)
        self.embedding.to_csv(self.args.embedding_output_path, index = None)

    def save_persona_graph_mapping(self):
        with open(self.args.persona_output_path, "w") as f:
           json.dump(self.egonet_splitter.personality_map, f)                     
