from typing import Any

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dgl.nn import EGATConv
from pytorch_lightning.core import LightningModule
from torchmetrics import Accuracy


class GraphEGAT(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        h_node_feats,
        h_edge_feats,
        num_heads,
    ) -> None:
        super().__init__()
        self.conv1 = EGATConv(
            in_node_feats, in_edge_feats, h_node_feats, h_edge_feats, num_heads
        )
        self.conv2 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )
        self.conv3 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )
        self.conv4 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )

    def forward(self, graph, node_feature, edge_feature):
        h_node, h_edge = self.conv1(graph, node_feature, edge_feature)
        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)
        h_node, h_edge = self.conv2(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )

        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)
        h_node, h_edge = self.conv3(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )

        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)
        h_node, h_edge = self.conv4(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )

        return h_node.flatten(1, 2), h_edge.flatten(1, 2)


class MLPPredictorFromNode(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        # linear_2 ( relu ( linear_1 ( [h_u, h_v] ) ) )
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class MLPPredictorFromEdge(nn.Module):
    def __init__(self, h_edge_feats):
        super().__init__()
        self.W1 = nn.Linear(h_edge_feats, h_edge_feats)
        self.W2 = nn.Linear(h_edge_feats, 1)

    def apply_edges(self, edges):
        # linear_2 ( relu ( linear_1 ( [h_u, h_v] ) ) )
        h = edges.data["h"]
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.edata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class MLPPredictor(nn.Module):
    def __init__(self, h_node_feats, h_edge_feats, n_classes):
        super().__init__()
        self.W1 = nn.Linear(h_node_feats * 2 + h_edge_feats, h_edge_feats)
        self.W2 = nn.Linear(h_edge_feats, n_classes)

    def apply_edges(self, edges):
        # linear_2 ( relu ( linear_1 ( [h_u, h_v] ) ) )
        h = torch.cat([edges.src["h"], edges.dst["h"], edges.data["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h)))}

    def forward(self, g, h_nodes, h_edges):
        with g.local_scope():
            g.ndata["h"] = h_nodes
            g.edata["h"] = h_edges
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class GraphEGAT_EdgePred(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        h_node_feats,
        h_edge_feats,
        num_heads,
    ) -> None:
        super().__init__()
        self.gnn = GraphEGAT(
            in_node_feats, in_edge_feats, h_node_feats, h_edge_feats, num_heads
        )
        self.epred = MLPPredictor(
            h_node_feats * num_heads, h_edge_feats * num_heads, n_classes=7
        )

    def forward(self, graph, node_feature, edge_feature):
        self.h_node, self.h_edge = self.gnn.forward(
            graph, node_feature, edge_feature
        )

    def predict(self, subgraph, h_node, h_edge):
        return self.epred.forward(subgraph, h_node, h_edge)


class GNNTrainer(LightningModule):
    def __init__(
        self,
        in_node_feats: int = 184,
        in_edge_feats: int = 7,
        h_node_feats: int = 32,
        h_edge_feats: int = 32,
        num_heads: int = 4,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = GraphEGAT_EdgePred(
            in_node_feats, in_edge_feats, h_node_feats, h_edge_feats, num_heads
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=in_edge_feats)
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.norel_idx = in_edge_feats - 1

    def forward(self, graph, node_feature, edge_feature):
        self.model.forward(graph, node_feature, edge_feature)

    def training_step(self, batch, batch_idx):
        train_g = batch
        if train_g.num_edges() > 20:
            train_g = dgl.DropEdge(p=0.05)(train_g)
        num_edges = train_g.num_edges()
        train_g = dgl.add_self_loop(train_g, fill_data=self.norel_idx)
        num_added_edges = train_g.num_edges() - num_edges

        u, v = train_g.edges()
        train_g_pred = dgl.graph(
            (u[:-num_added_edges], v[:-num_added_edges]),
            num_nodes=train_g.num_nodes(),
        )

        self.model.forward(
            train_g,
            F.one_hot(
                train_g.ndata["label"], num_classes=self.in_node_feats
            ).float(),
            F.one_hot(
                train_g.edata["label"], num_classes=self.in_edge_feats
            ).float()
            * train_g.edata["weight"].unsqueeze(-1).float(),
        )

        edge_preds = self.model.predict(
            train_g_pred,
            self.model.h_node,
            self.model.h_edge[:-num_added_edges],
        )

        train_g.edata["label"][
            train_g.edata["mst"].squeeze() == 0
        ] = self.norel_idx
        labels = train_g.edata["label"][:-num_added_edges]

        loss = F.cross_entropy(edge_preds, labels)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

        accuracy = self.accuracy(edge_preds, labels)
        self.log(
            "train_acc",
            accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        seq_acc = accuracy == 1.0
        self.log(
            "train_seq_acc",
            seq_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def eval_step(self, batch, batch_idx, prefix: str):
        train_g = batch
        num_edges = train_g.num_edges()
        train_g = dgl.add_self_loop(train_g, fill_data=self.norel_idx)
        num_added_edges = train_g.num_edges() - num_edges

        u, v = train_g.edges()
        train_g_pred = dgl.graph(
            (u[:-num_added_edges], v[:-num_added_edges]),
            num_nodes=train_g.num_nodes(),
        )

        self.model.forward(
            train_g,
            F.one_hot(
                train_g.ndata["label"], num_classes=self.in_node_feats
            ).float(),
            F.one_hot(
                train_g.edata["label"], num_classes=self.in_edge_feats
            ).float()
            * train_g.edata["weight"].unsqueeze(-1).float(),
        )

        edge_preds = self.model.predict(
            train_g_pred,
            self.model.h_node,
            self.model.h_edge[:-num_added_edges],
        )

        train_g.edata["label"][
            train_g.edata["mst"].squeeze() == 0
        ] = self.norel_idx
        labels = train_g.edata["label"][:-num_added_edges]

        loss = F.cross_entropy(edge_preds, labels)
        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        accuracy = self.accuracy(edge_preds, labels)
        self.log(
            f"{prefix}_acc",
            accuracy,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        seq_acc = accuracy == 1.0
        self.log(
            f"{prefix}_seq_acc",
            seq_acc,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.1 ** (epoch // 20)
        )
        return [optimizer], [scheduler]
