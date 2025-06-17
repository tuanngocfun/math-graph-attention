import dgl
import torch.nn.functional as F

from data import load_graph_json
from model import GNNTrainer


def mst_prediction(train_g, gnn_model, num_syms=184, num_rels=7):
    num_edges = train_g.num_edges()
    train_g = dgl.add_self_loop(train_g, fill_data=num_rels - 1)
    num_added_edges = train_g.num_edges() - num_edges

    u, v = train_g.edges()

    train_g_pred = dgl.graph(
        (u[:-num_added_edges], v[:-num_added_edges]),
        num_nodes=train_g.num_nodes(),
    )

    gnn_model.forward(
        train_g,
        F.one_hot(train_g.ndata["label"], num_classes=num_syms).float(),
        F.one_hot(train_g.edata["label"], num_classes=num_rels).float()
        * train_g.edata["weight"].unsqueeze(-1).float(),
    )

    edge_preds = gnn_model.predict(
        train_g_pred,
        gnn_model.h_node,
        gnn_model.h_edge[:-num_added_edges],
    )

    mst_edges = edge_preds.argmax(1) != 6

    return mst_edges.numpy()


def write_dot(G, output_fn):
    dot_str = []
    dot_str += [
        "strict digraph G {",
        "    rankdir=LR;",
        "    node [shape=record, width=.1]",
    ]
    dot_str += [
        f"    {node} [ label=\"{G.nodes()[node]['label']}\"];"
        for node in G.nodes()
    ]

    for edge in G.edges():
        rel, prob = (
            G.get_edge_data(*edge)[0]["label"],
            G.get_edge_data(*edge)[0]["weight"],
        )

        if rel in ["Ab", "Sup"]:
            dot_str += [
                f'    {edge[0]}:ne -> {edge[1]} [label="{rel}", prob={prob}];'
            ]
        elif rel in ["Be", "In", "Sub"]:
            dot_str += [
                f'    {edge[0]}:se -> {edge[1]} [label="{rel}", prob={prob}];'
            ]
        elif rel in ["R", "No"]:
            dot_str += [
                f'    {edge[0]} -> {edge[1]} [label="{rel}", weight=2, prob={prob}];'
            ]
    dot_str += ["}"]

    with open(output_fn, "w") as f:
        f.writelines([line + "\n" for line in dot_str])


if __name__ == "__main__":
    import sys

    model = GNNTrainer().load_from_checkpoint(
        "checkpoint/egat_ilabo2021/lightning_logs/version_1/checkpoints/epoch=24-val_acc=0.9897.ckpt"
    )
    # .load_from_checkpoint('checkpoint/egat_crohme/lightning_logs/version_15/checkpoints/epoch=21-val_acc=0.9542.ckpt')
    model.cpu()
    import json

    import numpy as np

    input_json = sys.argv[1]
    g, nodes_2_ids, relations = load_graph_json(input_json)
    mst_edges_pred = mst_prediction(train_g=g, gnn_model=model.model)
    with open(input_json, "w") as f:
        f.write(
            json.dumps(
                [relations[i] for i in np.where(mst_edges_pred == True)[0]]
            )
        )
