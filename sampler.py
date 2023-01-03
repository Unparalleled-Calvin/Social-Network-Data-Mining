import easygraph as eg
import networkx as nx

from toolset import class_transform


class Sampler:
    def __init__(self, graph: eg.Graph):
        self.graph = graph
        
    def sample(self, sampler_class, ratio = 0.01, max_num = 500, min_num = 0, save_file = True, file_name = False):
        num = max(min(int(ratio * len(self.graph.nodes)), max_num), min_num)
        print(f"start to sample with {sampler_class.__name__}, {num} nodes in total")
        sampler = sampler_class(number_of_nodes=num)
        G_, index_of_node, node_of_index = self.graph.to_index_node_graph()
        subgraph_ = sampler.sample(class_transform(G_, nx.Graph))
        nodes = [node_of_index[i] for i in subgraph_.nodes]
        subgraph = self.graph.nodes_subgraph(nodes)
        if save_file:
            if not file_name:
                file_name = f"{sampler_class.__name__}_subgraph.gexf"
            eg.write_gexf(subgraph, file_name)
        return subgraph
    
    def diffusion(self, ratio = 0.01, max_num = 500, min_num = 0, save_file = True, file_name = False): # diffusion
        from littleballoffur import DiffusionSampler
        return self.sample(DiffusionSampler, ratio=ratio, max_num=max_num, min_num=min_num, save_file=save_file, file_name=file_name)

    def diffusion_tree(self, ratio = 0.01, max_num = 500, min_num = 0, save_file = True, file_name = False): # diffusion tree
        from littleballoffur import DiffusionTreeSampler
        return self.sample(DiffusionTreeSampler, ratio=ratio, max_num=max_num, min_num=min_num, save_file=save_file, file_name=file_name)

    def forest_fire(self, ratio = 0.01, max_num = 500, min_num = 0, save_file = True, file_name = False):
        from littleballoffur import ForestFireSampler
        return self.sample(ForestFireSampler, ratio=ratio, max_num=max_num, min_num=min_num, save_file=save_file, file_name=file_name)

    def common_neighbor_aware_random_walk(self, ratio = 0.01, max_num = 500, min_num = 0, save_file = True, file_name = False):
        from littleballoffur import CommonNeighborAwareRandomWalkSampler
        return self.sample(CommonNeighborAwareRandomWalkSampler, ratio=ratio, max_num=max_num, min_num=min_num, save_file=save_file, file_name=file_name)
