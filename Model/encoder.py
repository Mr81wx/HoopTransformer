import torch
import torch.nn as nn

from Model.utils import *

from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, device, in_feat_dim,  time_steps=121, feature_dim=256, 
    head_num=4,  k=4): 
        super().__init__()
        self.device = device
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        #self.max_dynamic_rg = max_dynamic_rg        # GD
        #self.max_static_rg = max_static_rg          # GS
        assert feature_dim % head_num == 0      
        self.head_dim = int(feature_dim/head_num)   # d
        self.k = k                                  # k
        self.embedding_dim = 12
        
        # Initialize players, ball embeddings.
        #initrange = 0.1
        #n_player_ids = 500
        #self.player_embedding = nn.Embedding(n_player_ids, self.embedding_dim)
        #self.player_embedding.weight.data.uniform_(-initrange, initrange)

        #self.ball_embedding = nn.Parameter(torch.Tensor(self.embedding_dim))
        #nn.init.uniform_(self.ball_embedding, -initrange, initrange)
        
        #in_feat_dim = in_feat_dim + self.embedding_dim
        
        # TODO: replace custom multihead attention layer to nn.MultiHeadAttention
        # layer A : input -> [A,T,in_feat_dim=16] / output -> [A,T,D]
        self.layer_A = nn.Sequential(nn.Linear(in_feat_dim,32), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(32), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(32,128), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(128), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(128,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU() )
        self.layer_A.apply(init_xavier_glorot)
        
        # layer B : input -> [GD,T,in_dynamic_rg_dim] / output -> [GD,T,D]
        '''self.layer_B = nn.Sequential(nn.Linear(in_dynamic_rg_dim,32), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(32), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(32,128), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(128), Permute4Batchnorm((0,2,1)), nn.ReLU(),
                            nn.Linear(128,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU())
        '''
        #self.layer_B.apply(init_xavier_glorot)
        # layer C : input -> [GD,T,in_dynamic_rg_dim] / output -> [GD,T,D]
        '''self.layer_C = nn.Sequential(nn.Linear(in_static_rg_dim,32), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(32), Permute4Batchnorm((0,2,1)), nn.ReLU(), 
                            nn.Linear(32,128), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(128), Permute4Batchnorm((0,2,1)), nn.ReLU(),
                            nn.Linear(128,feature_dim), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)), nn.ReLU())
        '''
        #self.layer_C.apply(init_xavier_glorot)
        
        #Positional embedding
        self.layer_B = PositionalEncoding(self.feature_dim,0.1,self.time_steps)
        
        # layer D,E,F,G,H,I : input -> [A,T,D] / outpu -> [A,T,D]
        self.layer_D = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        
        self.layer_E = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)
        self.layer_F = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        self.layer_G = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)
        self.layer_H = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        self.layer_I = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)

        #self.layer_J = CrossAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k)
        #self.layer_K = CrossAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k)

        self.layer_L = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        self.layer_M = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)

        #self.layer_N = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        #self.layer_O = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)

        self.layer_P = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        self.layer_Q = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)

        #self.layer_DH = nn.Sequential(nn.Linear(self.feature_dim,self.feature_dim*4), Permute4Batchnorm((0,2,1)),
                            #nn.BatchNorm1d(self.feature_dim*4), Permute4Batchnorm((0,2,1)), nn.ReLU())
        #self.layer_DH.apply(init_xavier_glorot)
        

    def forward(self, state_feat, agent_batch_mask, padding_mask, hidden_mask, agent_ids_batch):
        #print(state_feat.shape)
        state_feat = state_feat.clone()
        state_feat[hidden_mask] = -1
        
        #cat player embedding
        device = torch.device("cuda")
        '''ball_indices = torch.where(agent_ids_batch == -1)[0].tolist()
        player_indices = torch.where(agent_ids_batch != -1)[0].tolist()

        ball_embedded = self.ball_embedding.repeat(len(ball_indices), 1) # ball embedding
        player_ids = agent_ids_batch[player_indices].long()
        player_embedded = self.player_embedding(player_ids.flatten().to(device)) # player embedding
        
        
        merged_tensor = torch.empty((len(agent_ids_batch), self.embedding_dim), device=device)
        
        merged_tensor[ball_indices, :] = ball_embedded
        merged_tensor[player_indices, :] = player_embedded
        
        embedded_expanded = merged_tensor.unsqueeze(1).repeat(1, self.time_steps, 1)
        

        embedded_state_feat = torch.cat((state_feat, embedded_expanded), dim=2)
                '''

        A_ = self.layer_A(state_feat)
        #B_ = self.layer_B(traffic_light_feat)
        #C_ = self.layer_C(road_feat)
        
        output = self.layer_B(A_)
        output = self.layer_D(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_E(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_F(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_G(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_H(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_I(output,agent_batch_mask, padding_mask, hidden_mask)

        # TODO : add additional artificial agent/time AND adjust mask for it

        #output, att_weight_J = self.layer_J(output,C_,agent_rg_mask, roadgraph_valid, hidden_mask)
        #output, att_weight_K = self.layer_K(output,B_,agent_traffic_mask, traffic_light_valid, hidden_mask)

        output = self.layer_L(output,agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_M(output,agent_batch_mask, padding_mask, hidden_mask)

        #output = self.layer_N(output,agent_batch_mask, padding_mask, hidden_mask)
        #output = self.layer_O(output,agent_batch_mask, padding_mask, hidden_mask)

        output = self.layer_P(output,agent_batch_mask, padding_mask, hidden_mask)
        Q_ = self.layer_Q(output,agent_batch_mask, padding_mask, hidden_mask)
        #Q_ = self.layer_DH(Q_)
        return {'out': Q_, 'att_weights': 0}

