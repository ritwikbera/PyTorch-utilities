#average across all dimensions
class InstanceNorm(nn.Module):
	def __init__(self, *args, **kwargs):
       super(InstanceNorm, self).__init__()

    def forward(self, x):
        x2 = x.view(*x.shape[:2], -1)
        mean = x2.mean(-1)
        var = x2.var(-1)
        mean = mean.view(*mean.shape[:2], *((len(x.shape) - 2) * [1]))
        var = var.view(*var.shape[:2], *((len(x.shape) - 2) * [1]))
        x = (x - mean)/(var + 1e-6).sqrt()
        return x

#doesn't average across channel dim
class BatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        super(BatchNorm2D, self).__init__(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x):
        exponential_average_factor = 0.0
        if self.training:
            exponential_average_factor = self.momentum

        if self.training:
            mean = x.mean([0,2,3])
            var = x.var([0,2,3], unbiased=False)
            n = x.numel()/x.size(1)
            self.running_mean = exponential_average_factor*self.running_mean + (1 - exponential_average_factor)*mean
            self.running_var = exponential_average_factor*self.running_var + (1 - exponential_average_factor)*(n/(n-1))*var
        else:
            mean = self.running_mean
            var = self.running_var 

        x = (x - mean[None,:,None,None]) / (torch.sqrt(var[None,:,None,None]+self.eps))
        if self.affine:
            x = x*self.weight[None,:,None,None] +self.bias[None,:,None,None] 

        return x 

class VisualAttention(nn.Module):
    def __init__(self, encoded_img_dim, decoder_dim, attention_dim):
        super(VisualAttention, self).__init__()
        self.encoder_att = nn.Linear(encoded_img_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        #decoder output spread over each pixel, scalar attention output dim squeezed
        att = self.full_att(self.relu(att1+att2.unsqueeze(1))).squeeze(2) 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    #call at start of every sequence processing event
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden