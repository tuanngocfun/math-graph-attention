# Link prediction by Graph Attention Networks
Multi-layers of Graph Attention Networks combine node features (symbols) and edge features (relations) to predict if a relation connection should be removed or not.

## Overview

The model use primitive graphs (directed graph of symbols and relations) as input and output the Symbol Relation Tree (a type of minimum spanning tree) of that graph.

Primitive graphs are created by a recognizer which recognize symbols and build up pairwise relations by the recognizer and spatial search.

Link prediction run a predictor in the output graph and predict if an edge should be retained or not.

## Models
The model consist of a graph neural network (edge graph attention network - EGAT) and a link predictor.

- Input data: 

Node data: 

A numpy array of symbol labels (indexes)
```
g.ndata["label"] 
```

Edge data: 

A numpy array of relation labels (indexes)

```
g.edata["label"] 
```
A numpy array of relation probabilities
```
g.edata["weight"]
```
- Label data (for training edge prediction):

A numpy array (boolean) representing an edge is belong to the Symbol Relation Tree or not.

```
g.edata["mst"]
```

## Training data preparation
Create pickle data from primitive graph (json) files:


From [data.py](data.py)
```
math_graph_ds.create_data(input_jsons, input_lgs)
pickle.dump(math_graph_ds.graphs, open(output_pickle, "wb"))
```

## Run training
Set dataset paths in [data.py](data.py)


```
class MathGraphData(pl.LightningDataModule):
    def __init__(
        ...
        train_data: str = "ilabo2021.pkl",
        val_data: str = "test2016.pkl",
        test_data: str = "test2019.pkl",
```

Run [trainer_lt.py](trainer_lt.py)