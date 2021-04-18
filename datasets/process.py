import gzip
from tqdm import tqdm
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import gensim


b_type_dict = {
    "also_viewed": 0,
    "also_bought": 0,
    "bought_together": 0,
    "buy_after_viewing": 0
}

_urls = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
cat = "Video_Games"
node_feat = 300


class AmazonDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AmazonDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["meta_" + cat + ".json.gz"]

    @property
    def processed_file_names(self):
        return 'Amazon_%s.pt' % cat

    def download(self):
        raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' 
            % (_urls + "meta_" + cat + ".json.gz", self.raw_dir))

    def parse(self, path):
      g = gzip.open(path, 'rb')
      for l in g:
        yield eval(l)

    def getDF(self, path):
      i = 0
      df = {}
      print("Reading data...")
      for d in tqdm(self.parse(path)):
        df[i] = d
        i += 1
      return pd.DataFrame.from_dict(df, orient='index')

    def getDict(self, df):
        d = {}
        for idx, raw in df.items():
            d[raw] = idx
        return d

    def amazon_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            feat.append((n, d['node_feat']))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def amazon_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(x in d["b_type"]) for x in sorted(list(b_type_dict.keys()))]
            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))

        return edge_index, edge_attr

    def process(self):
        data = self.getDF(self.raw_paths[0])
        node_idx_dict = self.getDict(data.asin)
        d2v_model = gensim.models.doc2vec.Doc2Vec.load(self.raw_dir + "/reviews_" + cat + ".d2v")
        nodes = set(node_idx_dict.keys()) & set(d2v_model.docvecs.doctags.keys())

        graph = nx.DiGraph()
        print("Constructing graph...")

        for idx, item in tqdm(data.iterrows(), total=data.shape[0]):

            # add nodes and edges
            if item["asin"] in nodes:
                graph.add_node(node_idx_dict[item["asin"]], node_feat=d2v_model.docvecs[item["asin"]], asin=item["asin"], title=item["title"])
                if not pd.isna(item["related"]):
                    relations = item["related"]
                    for b_type in relations.keys():
                        b_type_dict[b_type] += len(relations[b_type])
                        for dest in relations[b_type]:
                            if dest in nodes:
                                if (node_idx_dict[item["asin"]], node_idx_dict[dest]) in graph.edges:
                                    graph.edges[node_idx_dict[item["asin"]], node_idx_dict[dest]]["b_type"].append(b_type)
                                else:
                                    graph.add_edge(node_idx_dict[item["asin"]], node_idx_dict[dest], b_type=[b_type])

        # remove all isolates
        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)
        graph = nx.convert_node_labels_to_integers(graph)

        node_attr = self.amazon_nodes(graph)
        edge_index, edge_attr = self.amazon_edges(graph)

        print("Graph completed!")

        amazon_data = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        )

        data_list = [amazon_data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    root = "../data/"
    dataset = AmazonDataset(root=root)
    dataset.process()
    




