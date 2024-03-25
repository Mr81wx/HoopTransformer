import math
from torch.distributions.laplace import Laplace
import torch
import torch.nn as nn
#from visualizer import get_local
import functools

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Permute4Batchnorm(nn.Module):
    def __init__(self,order):
        super(Permute4Batchnorm, self).__init__()
        self.order = order
    
    def forward(self, x):
        assert len(self.order) == len(x.shape)
        return x.permute(self.order)

class ScaleLayer(nn.Module):

   def __init__(self, shape, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

   def forward(self, input):
       return input * self.scale

def init_xavier_glorot(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class SelfAttLayer_Enc(nn.Module):
    def __init__(self, time_steps=121, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num,add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)
         
    #@get_local('attention_map')
    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        #print(hidden_mask)
        A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A_,A__ = batch_mask.shape
        assert (A==A_ and A==A__)
        A___,T_ = padding_mask.shape
        assert (A==A___ and T==T_)

        x_ = self.layer_X_(x)                               # [A,T,D]

        if self.across_time:
            q_ = x_.permute(1,0,2)                          # [L,N,E] : [A,T,D]->[T,A,D]
            k,v = x_.permute(1,0,2), x_.permute(1,0,2)      # [S,N,E] : [A,T,D]->[T,A,D]

            key_padding_mask = padding_mask                 # [N,S] : [A,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,A,D]
            att_output, att_weight = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [A,T,D]
            
            # add attention map
            attention_map = att_weight
            
            att_output = att_output.permute(1,0,2)
        else:
            q_ = x_                                         # [L,N,E] = [A,T,D]
            k, v = x_, x_                                   # [S,N,E] = [A,T,D]

            key_padding_mask = padding_mask.permute(1,0)    # [N,S] = [T,A]
            attn_mask = batch_mask                          # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T,D]
            att_output, att_weight = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # add attention map
            attention_map = att_weight
            
        
        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_

class SelfAttLayer_Dec(nn.Module):
    def __init__(self, time_steps=121, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num,add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        F,A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A_,A__ = batch_mask.shape
        assert (A==A_ and A==A__)
        # A___,T_ = padding_mask.shape
        # assert (A==A___ and T==T_)

        x_ = self.layer_X_(x)                                           # [F,A,T,D]

        if self.across_time:
            q = x_.reshape((-1,T,D)).permute(1,0,2)                     # [L,N,E] : [F,A,T,D]->[F*A,T,D]->[T,F*A,D]
            k,v = q, q                                                  # [S,N,E] : [T,F*A,D]

            key_padding_mask = None#padding_mask.repeat(F,1)                 # [N,S] : [A*F,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,F*A,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((T,F,A,D)).permute(1,2,0,3)
        else:
            q = x_.permute(0,2,1,3).reshape((-1,A,D)).permute(1,0,2)    # [L,N,E] : [F,A,T,D]->[F,T,A,D]->[F*T,A,D]->[A,T*F,D]
            k, v = q, q                                                 # [S,N,E] : [A,T*F,D]

            key_padding_mask = None#padding_mask.permute(1,0).repeat(F,1)    # [N,S] = [T*F,A]
            attn_mask = batch_mask                                      # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T*F,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((A,F,T,D)).permute(1,0,2,3)

        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_



class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=121):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(0).unsqueeze(-1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        


def laplace_kl(gt,predict,target_scale):
    #gt [A*T,2] x,y predict[A*T,F,4] x.loc,x.scale,y,loc,y.scale
    F = predict.size(1)
    gt = gt.unsqueeze(1).repeat(1, F, 1)
    gt_scale = torch.ones_like(gt).fill_(target_scale) #创建一个相同shape的gt_scale
    
    gt_dist = Laplace(gt, gt_scale)
    predict_dist = Laplace(predict[:, :, 0::2], predict[:, :,1::2])
    kl_div = torch.distributions.kl.kl_divergence(gt_dist, predict_dist) #[A*T,F,2]
    return kl_div

def team_pooling(out):
    #print(out.shape)
    player = out[:,1:,:,:]
    ball = out[:,0,:,:]

    player_max, _ = torch.max(player,dim=1)
    player_ball = torch.stack([player_max, ball],dim=1)
    #print(player_ball.shape)
    mean_tensor = torch.mean(player_ball,dim=[1,2])
    
    return mean_tensor
    
def player_pooling(out):
    player = out[:,1:,:,:]
    player_pooling = torch.mean(player,dim = [2])
    return player_pooling
