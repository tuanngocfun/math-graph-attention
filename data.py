import json
import os
import pickle
from glob import glob
from typing import Optional

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from parse_lg import *
from parse_lg import parse_lg


class SymbolCandidate:
    __slots__ = ("latex", "strokeIds")

    def __init__(self, *args):
        if len(args) == 2:
            self.latex = args[0]
            self.strokeIds = args[1]

    def toJson(self):
        return {"latex": self.latex, "strokeIds": self.strokeIds}


class Relation:
    __slots__ = ("prob", "relation", "symChild", "symRoot")

    def __init__(self, *args):
        if len(args) == 4:
            self.symRoot = args[0]
            self.symChild = args[1]
            self.relation = args[2]
            self.prob = args[3]

    def toJson(self):
        return {
            "prob": self.prob,
            "relation": self.relation,
            "symChild": self.symChild.toJson(),
            "symRoot": self.symRoot.toJson(),
        }


vocab = [line.strip() for line in open("data/HandsCTC.lbl").readlines()]
sym_vocab = vocab[:-7]
rel_vocab = vocab[-7:-1]


def load_graph_json(input_json):
    try:
        relations = json.load(open(input_json))
    except:
        return None, None, None
    if len(relations) == 0:
        return None, None, None

    sym2Latex = {}
    for rel in relations:
        sym2Latex.update(
            {str(rel["symChild"]["strokeIds"]): rel["symChild"]["latex"]}
        )
        sym2Latex.update(
            {str(rel["symRoot"]["strokeIds"]): rel["symRoot"]["latex"]}
        )

    edges = [
        (
            str(rel["symRoot"]["strokeIds"]),
            str(rel["symChild"]["strokeIds"]),
            {"weight": rel["prob"], "relation": rel["relation"]},
        )
        for rel in relations
    ]
    from_nodes, to_nodes = list(zip(*[(edge[0], edge[1]) for edge in edges]))

    import networkx as nx

    G = nx.MultiDiGraph()
    G.add_edges_from(
        [
            (
                str(rel["symRoot"]["strokeIds"]),
                str(rel["symChild"]["strokeIds"]),
                {"weight": rel["prob"], "relation": rel["relation"]},
            )
            for rel in relations
        ]
    )
    for node_id in G.nodes:
        G.nodes[node_id]["label"] = sym2Latex[node_id]

    nodes_2_ids = {node: id for id, node in enumerate(G.nodes)}

    g = dgl.graph(
        (
            [nodes_2_ids[node] for node in from_nodes],
            [nodes_2_ids[node] for node in to_nodes],
        ),
    )

    g.ndata["label"] = torch.from_numpy(
        np.array([sym_vocab.index(sym2Latex[node_id]) for node_id in G.nodes])
    )
    g.edata["label"] = torch.from_numpy(
        np.array([rel_vocab.index(edge[2]["relation"]) for edge in edges])
    )
    g.edata["weight"] = torch.from_numpy(
        np.array([(edge[2]["weight"]) for edge in edges])
    )

    return g, nodes_2_ids, relations


def create_graph_json(input_json, input_lg):
    g, nodes_2_ids, _ = load_graph_json(input_json)

    if g != None and nodes_2_ids != None:
        objs, rels = parse_lg(input_lg)
        id2gid = {
            obj.id: nodes_2_ids.get(str(obj.strokes), -1) for obj in objs
        }
        if -1 in id2gid.values():
            print(f"check {input_json} {input_lg}")
            return None

        g.edata["mst"] = torch.zeros(g.num_edges(), 1)

        for rel in rels:
            if (
                id2gid.get(rel.id1, "") != ""
                and id2gid.get(rel.id2, "") != ""
                and g.has_edges_between(id2gid[rel.id1], id2gid[rel.id2])
            ):
                g.edata["mst"][
                    g.edge_ids(id2gid[rel.id1], id2gid[rel.id2])
                ] = 1

    return g


class MathGraph(DGLDataset):
    def __init__(self):
        self.graphs = []
        super().__init__(name="crohme2019")

    def process(self):
        return

    def create_data(self, input_jsons, input_lgs):
        for fn in glob(f"{input_jsons}/*.json"):
            fn_lg = os.path.join(input_lgs, os.path.basename(fn).replace(".json", ".lg"))
            g = create_graph_json(fn, fn_lg)
            if g != None:
                self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


def get_dataset(pkl_fn):
    graph_data = pickle.load(open(pkl_fn, "rb"))
    graph_ds = MathGraph()
    graph_ds.graphs = graph_data
    return graph_ds


import pytorch_lightning as pl


class MathGraphData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        workers: int = 1,
        train_data: str = "train.pkl",
        val_data: str = "test2014.pkl",
        test_data: str = "test2019.pkl",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = get_dataset(self.train_data)
            self.val_dataset = get_dataset(self.val_data)
        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(self.test_data)

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(
            torch.arange(len(self.train_dataset))
        )
        train_loader = GraphDataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=False,
        )
        return train_loader

    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(torch.arange(len(self.val_dataset)))
        val_loader = GraphDataLoader(
            self.val_dataset,
            sampler=val_sampler,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        val_sampler = SubsetRandomSampler(torch.arange(len(self.test_dataset)))
        val_loader = GraphDataLoader(
            self.test_dataset,
            sampler=val_sampler,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return val_loader


if __name__ == "__main__":
    import pickle

    math_graph_ds = MathGraph()

    # math_graph_ds.graphs = []
    # input_jsons = "data/crohme_2022/Test2019_primitive_json"
    # input_lgs = "data/WithGT/LGs/Crohme_all"
    # math_graph_ds.create_data(input_jsons, input_lgs)
    # pickle.dump(math_graph_ds.graphs, open('test2019.pkl', 'wb'))

    math_graph_ds.graphs = []
    input_jsons = "data/Test2014_primitive_json"
    input_lgs = "data/Crohme_all_LGs"
    math_graph_ds.create_data(input_jsons, input_lgs)
    pickle.dump(math_graph_ds.graphs, open("test2014.pkl", "wb"))
