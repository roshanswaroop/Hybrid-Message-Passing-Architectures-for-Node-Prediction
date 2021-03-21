import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

# imported for 224W project
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch.nn import Parameter
import torch_scatter
import torch_geometric


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GNNHybrid(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GNNHybrid, self).__init__()

        # below code is very similar to GCN & GraphSage: real difference is in GNNFilMConv layer
        # change the code for hybrid layer ordering

        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNFiLMConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        print(self.convs)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GNNFiLM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GNNFiLM, self).__init__()

        # below code is very similar to GCN & GraphSage: real difference is in GNNFilMConv layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNFiLMConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GNNFiLMConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GNNFiLMConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GNNFiLMConv(MessagePassing):
    r""" Inspired by paper;
    uses format from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html
    """

    def __init__(self, in_channels: int, out_channels: int,
                 normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GNNFiLMConv, self).__init__(**kwargs)

        # uses learnable function to compute params beta, gamma for transform
        # Layer to compute the message from a source state
        # code inspiration: https://github.com/microsoft/tf-gnn-samples/blob/a78f47566c7974c1cf1494510ee5260bcd73922a/gnns/gnn_film.py#L8
        # Layer to compute the \beta/\gamma weights for FiLM
        self.edge_type_to_film_computation_layers = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        self.lin_layer = torch.nn.Linear(in_features=out_channels, out_features=out_channels, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
   
        self.normalize = normalize
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = self.lin_layer(out)

        if (self.normalize):
        	out = torch.nn.functional.normalize(out)
        return out

    def reset_parameters(self):
        self.edge_type_to_film_computation_layers.reset_parameters()
        self.lin_layer.reset_parameters()

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
    	return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    	# aggregation function is sum
    	#out = torch_scatter.scatter(src=x, index=adj_t, dim_size=x.size(0), reduce='sum')
    	# step 1: apply gamma, beta to curr_emb; and then get activation function
    	# activation function is LeakyReLU
    	neigh_emb = matmul(adj_t, x, reduce=self.aggr)
    	#print(neigh_emb.shape)
    	w_emb = self.edge_type_to_film_computation_layers(neigh_emb)
    	#print(w_emb.shape)

    	leaky_relu = torch.nn.LeakyReLU()
    	out = leaky_relu(w_emb)
    	#print(out.shape)

    	return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)

    # Added to code for 224W project
    parser.add_argument('--use_film', action='store_true')
    parser.add_argument('--use_hybrid', action='store_true')
    parser.add_argument('--use_gat', action='store_true')


    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.use_film:
    	print("using GNNFiLM")
    	model = GNNFiLM(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.use_hybrid:
    	print("using GNNHybrid")
    	model = GNNHybrid(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.use_gat:
    	print("using GAT")
    	model = GAT(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    print("initalized model")

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
