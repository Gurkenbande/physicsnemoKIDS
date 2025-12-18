import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch


class GNN4CD_Model(nn.Module):
    
    def __init__(self, encoding_dim=128, n_input_features=28, h_hid=32, n_layers=2, 
                 high_in=4, low2high_out=64, high_out=64, n_output_vars=4):
        
        super(GNN4CD_Model, self).__init__()
        self.n_output_vars = n_output_vars
        
        # for seq_l
        self.rnn = nn.GRU(input_size=n_input_features, hidden_size=h_hid, num_layers=n_layers, batch_first=True)


        # without seq_l
        self.encoder = nn.Sequential(nn.Linear(n_input_features, encoding_dim),nn.ReLU())
 


        self.dense = nn.Sequential(nn.Linear(h_hid, encoding_dim), nn.ReLU())
        self.downscaler = geometric_nn.Sequential('x, edge_index', [(GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 'x, edge_index -> x')])

        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU()])
        
        self.predictor = nn.Sequential(nn.Linear(high_out,high_out),nn.ReLU(),nn.Linear(high_out,32),nn.ReLU(),nn.Linear(32,n_output_vars))
        
   

    def forward(self, data, inference=False):

        x_low = data.x_dict['low']
        if x_low.dim() == 2:
            encod_low = self.encoder(x_low)

        elif x_low.dim() == 3:
            # GRU expects [batch, seq, feature]
            encod_rnn, h = self.rnn(x_low)
            encod_low = self.dense(h[-1])

        else:
            raise ValueError(f"Unexpected low.x shape: {x_low.shape}")

        # low → high
        encod_low2high = self.downscaler(
            (encod_low, data.x_dict['high']),
            data["low", "to", "high"].edge_index
        )

        # high ↔ high
        encod_high = self.processor(
            encod_low2high,
            data.edge_index_dict[('high','within','high')]
        )

        x_high = self.predictor(encod_high)

        if inference:
            return x_high

        return x_high

    

        
