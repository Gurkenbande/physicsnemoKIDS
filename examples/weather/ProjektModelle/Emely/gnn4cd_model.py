import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class GNN4CD_Model(nn.Module):
    
    def __init__(self, encoding_dim=128, n_input_features=26, h_hid=25, n_layers=2, 
                 high_in=4, low2high_out=64, high_out=64, n_output_vars=4):
        
        super(GNN4CD_Model, self).__init__()
        self.n_output_vars = n_output_vars
        
        self.rnn = nn.GRU(input_size=n_input_features, hidden_size=h_hid, 
                         num_layers=n_layers, batch_first=True)
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
        
   
   
    def forward(self, data, inference = False):
        encod_rnn, h = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = h[-1] 
        encod_rnn = self.dense(encod_rnn)
        encod_low2high = self.downscaler((encod_rnn, data.x_dict['high']), data["low", "to", "high"].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)
        '''
        if inference:
            data['high'].x_high = x_high
            return x_high'''

        return x_high
    
'''
    def forward(self, data):
            encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
            encod_rnn = encod_rnn.flatten(start_dim=1)
            encod_rnn = self.dense(encod_rnn)
            encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data["low", "to", "high"].edge_index)
            encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
            x_high = self.predictor(encod_high)

            return x_high'''

    
