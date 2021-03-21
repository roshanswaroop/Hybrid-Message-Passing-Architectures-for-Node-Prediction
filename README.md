# Hybrid-Message-Passing-Architectures-for-Node-Prediction
Hybrid Message-Passing Architectures for Node Prediction: uses OGB to explore techniques

# Exploring Hybrid Models with Feature-wise Linear Modulation

This project uses data from [Stanford OGB](https://ogb.stanford.edu/). Specifically, the [arxiv dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) for node property prediction.

The code in `gnn.py` is based on [OGB code](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py) and meant to be used in coordination with OGB sample code. In order to run the project, take the following steps:

1. Download `ogb` from https://github.com/snap-stanford/ogb, or run `git clone https://github.com/snap-stanford/ogb` to get it locally.
2. Go to the [arxiv directory](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv): `cd ogb/examples/nodeproppred/arxiv`
3. Download `gnn.py` from this repository. Replace the `gnn.py` in that directory with this version.
4. Run the code! To start, run `python gnn.py` to run the OGB baseline GCN, implemented by Stanford OGB project owners (not by the authors of this project).

The authors of this project implemented the following classes in `gnn.py`:
- GAT
- GNNHybrid
- GNNFiLM
- GNNFiLMConv
- GATConv

In order to run one of the models implemented by the authors, run the following commands:

Baseline GAT:

`python gnn.py --use_gat`

GNN FiLM (based on [this paper](https://arxiv.org/pdf/1906.12192.pdf) from ICML 2020):

`python gnn.py --use_film`

GNN Hybrid, composed of layers from different GNN types:

`python gnn.py --use_hybrid`

