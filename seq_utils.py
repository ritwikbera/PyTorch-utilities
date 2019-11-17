import torch
import torch.nn as nn

class Masked_Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(Masked_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
    
    def forward(self, inputs, mask)
        hidden_state = torch.zeros(inputs.size(0), self.hidden_dim)
        cell_state = torch.zeros(inputs.size(0), self.hidden_dim)
        
        outputs = []
        for step in range(inputs.size(1)):
            hidden = self.lstm_cell(inputs[:, step], (hidden_state, cell_state))
            hidden = (mask[:, step, None] * hidden[0],
                      mask[:, step, None] * hidden[1])  
            outputs.append(hidden[0])
            
        return torch.stack(outputs, dim=1)