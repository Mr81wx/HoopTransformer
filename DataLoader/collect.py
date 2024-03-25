import pandas as pd
import numpy as np
from .geometry import velocity,reversal_coord,split_list_by_v,create_random_tensor
import torch
import random


def create_batch_A(batch): #MP 
    
    sample_freq = 5 
    sequence_length = int(24*25/sample_freq + 1) # Time length
    list_24s = [i*(1/sample_freq) for i in range(0, sequence_length)] # 24s game clock
    list_24s.reverse() 

    #set time_steps = 121，set a states_batch to save input data，shape = [batch_size*agent, time_steps, 3]
    time_steps = sequence_length # 121
    states_batch = np.array([]).reshape(-1,time_steps,3)

    # states_padding_batch for padding ; states_hidden_BP_batch for hidden
    states_padding_batch = np.array([]).reshape(-1,time_steps) #shape [batch_size*agent, time_steps]
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps) #shape [batch_size*agent, time_steps]

    num_agents = np.array([]) 

    for scene in batch:
        
        scene_tensor = None
        for i in range(len(scene.agents)): # 共有11个agents
            agent = scene.agents[i]
            
            if agent_team != scene.def_teamid:
                single_agent_array = np.array([[x,y,v] for x,y,v in zip(agent.x,agent.y, agent.v)][::sample_freq]) # 按照sample_freq，即5帧/s取样
                single_agent_tensor = torch.Tensor(torch.tensor([single_agent_array.transpose()]))
                single_agent_tensor = single_agent_tensor.to(torch.float)
                # (time_steps, 3) transfer-> (3, time_steps); 3: x,y,velocity
            
                if scene_tensor == None:
                    scene_tensor = single_agent_tensor
                else:
                    scene_tensor = torch.cat([scene_tensor, single_agent_tensor], dim=0) #take 6 agents (ball+5 off player)tensor shape=[6,3,time_steps]
                
        scene_tensor = scene_tensor
        #padding agents length to 6, some possession dont have ball data
        A,D,T = scene_tensor.size()
        if A < 6:
            new_dim = torch.full((6-A, D, T), -1)
            scene_tensor = torch.cat([new_dim, scene_tensor], dim=0)
            #agent_ids = np.pad(agent_ids, (11 - A, 0), 'constant', constant_values=(-1,))
        
        #agent_ids_batch = np.append(agent_ids_batch,agent_ids)
        scene_tensor = torch.transpose(scene_tensor , dim0=1, dim1=2) #shape[A,T,D]
        time_24s = scene.time_24s[::sample_freq] #间隔sample_freq个取值

        # padding time_length to 121
        head_padding_size = 0
        end_padding_size =  121 - (head_padding_size+len(time_24s)+1)  #endtime～0

     
        # use 0 to padding 
        states_feat = torch.nn.functional.pad(scene_tensor,  (0, 0, head_padding_size+1, end_padding_size, 0, 0), mode='constant', value=0)
        states_padding = states_feat[:,:,0]
        states_padding = states_padding == 0

        
        #hidden #according Velocity masking
        
        states_hidden_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_) #True = hidden
        #Take 1/3 of the valid time_step
        split_time = len(time_24s) * 2 // 3 
        states_hidden_BP[:,:split_time] = False #visible = False, predinction=True
        
        num_agents = np.append(num_agents, len(states_feat)) # numpy array(batch_size,) [6，6，6，6，6，6]
        
        # 把每个batch的states_feat, states_padding, states_hidden_BP 合并成states_batch, states_padding_batch, states_hidden_BP_batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden_BP), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64) # 现在index=0处插入0，再进行累计求和 [0, 11, 22, ....]
    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) #创建一个全为1的[A*batch,A*batch]矩阵

    for i in range(len(num_agents)):
        # 构造(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    agent_ids_batch = torch.empty(1,1)
    # states_batch是(batch_size,11,121,4)的tensor，用于储存6个agents 的轨迹，末尾被补全,用于符合模型输入
    # agents_batch_mask 是(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵，便于之后attention focus
    # states_padding_batch 是储存padding的tensor,(batch_size, 11, 121)
    # states_hidden_BP_batch 是储存mask的tensor, (batch_size, 11, 121)
    # num_agents_accum 累计的agents的num，(1+batch_size, ) 累计的agents数量
    # agent_ids_batch 是agents id的tensor, (batch_size, 11), 如果不足6个agents，用-1补全了
    return (states_batch, agents_batch_mask, states_padding_batch, states_hidden_BP_batch, num_agents_accum, agent_ids_batch)
    
def create_batch_B(batch): #MR
    
    #tensor序列长度 11*121*3 #补齐T维度 到121
    sample_freq = 5 # 帧数
    sequence_length = int(24*25/sample_freq + 1) # 24s 对应的序列长度
    list_24s = [i*(1/sample_freq) for i in range(0, sequence_length)] # sequence_length 对应的game clock
    list_24s.reverse() # 反转，从24s 到0s 的变化

    #设置时间步长为序列长度，并初始化一个空的NumPy数组states_batch，用于存储状态信息，其形状将会是[batch_size, time_steps, 3]
    time_steps = sequence_length # 121
    states_batch = np.array([]).reshape(-1,time_steps,3)

    # 初始化两个空的NumPy数组，states_padding_batch和states_hidden_BP_batch，用于存储填充状态和隐藏状态
    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([]) #每一个poss的agent数量
    #agent_ids_batch = np.array([]) #每一个batch的agent ids, = batch size * 11
    
    #player_newids = pd.read_csv('players.csv')
    
    for scene in batch:
        
        scene_tensor = None
        #agent_ids = np.array([])
        for i in range(len(scene.agents)): # 共有11个agents
            agent = scene.agents[i]
            if agent_team != scene.def_teamid:
                single_agent_array = np.array([[x,y,v] for x,y,v in zip(agent.x,agent.y, agent.v)][::sample_freq]) # 按照sample_freq，即5帧/s取样
                single_agent_tensor = torch.Tensor(torch.tensor([single_agent_array.transpose()]))
                single_agent_tensor = single_agent_tensor.to(torch.float)
               # (time_steps, 4) 转置-> (4, time_steps); 4: x,y,v速率,t(ball/off/def)
            
                if scene_tensor == None:
                    scene_tensor = single_agent_tensor
                else:
                    scene_tensor = torch.cat([scene_tensor, single_agent_tensor], dim=0) #获得6个agent的tensor shape=[6,3,time_steps]
                
        scene_tensor = scene_tensor
        #补齐A的维度为6
        A,D,T = scene_tensor.size()
        # 如果agents 不足11个：用-1 补齐scence_tensor,agent_ids也用-1补齐
        if A < 6:
            new_dim = torch.full((6-A, D, T), -1)
            scene_tensor = torch.cat([new_dim, scene_tensor], dim=0)
            #agent_ids = np.pad(agent_ids, (11 - A, 0), 'constant', constant_values=(-1,))
        
        #agent_ids_batch = np.append(agent_ids_batch,agent_ids)
        scene_tensor = torch.transpose(scene_tensor , dim0=1, dim1=2) #shape[A,T,D]
        time_24s = scene.time_24s[::sample_freq] #间隔sample_freq个取值

        # 计算头部和尾部的填充padding大小
        head_padding_size = 0
        end_padding_size =  121 - (head_padding_size+len(time_24s)+1)  #endtime～0

        #[11,121,4] 用0来补齐 2023.04.01修改 #0414update, 顶端对齐
        # 对scence_tensor 的时间T维度用0进行padding
        # TODO：为什么要做前面padding一下即有head_padding_size+1，而不是head_padding_size
        states_feat = torch.nn.functional.pad(scene_tensor,  (0, 0, head_padding_size+1, end_padding_size, 0, 0), mode='constant', value=0)
        # states_padding: padding 的位置的tensor
        #bool型，(11,121) True为padding
        states_padding = states_feat[:,:,0]
        states_padding = states_padding == 0

        #
        #mask process #according Velocity masking
        v_tensor = scene_tensor[:,:,2] >= 4  #获得速度张量 [A,T,1]
        index_v = split_list_by_v(v_tensor)
        current_step = len(time_24s) #
        states_hidden_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_) #True为masked
        states_hidden_BP[:,:current_step+1] = False #实际轨迹为False
        #从index_v中的每一个agent中提取一段轨迹改为masked（True）
        for index,lst in enumerate(index_v):
            try:
                s,e = random.choice(lst)
            except:
                s,e = 0,0
            states_hidden_BP[index,s:e] = True
        states_hidden_BP[:,1:4] = False
        # print(states_hidden_BP.shape)
        
        num_agents = np.append(num_agents, len(states_feat)) # numpy array(batch_size,) [6，6，6，6，6，6]
        
        # 把每个batch的states_feat, states_padding, states_hidden_BP 合并成states_batch, states_padding_batch, states_hidden_BP_batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden_BP), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64) # 现在index=0处插入0，再进行累计求和 [0, 11, 22, ....]
    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) #创建一个全为1的[A*batch,A*batch]矩阵

    for i in range(len(num_agents)):
        # 构造(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    agent_ids_batch = torch.ones(1,1)
    # states_batch是(batch_size,11,121,4)的tensor，用于储存6个agents 的轨迹，末尾被补全,用于符合模型输入
    # agents_batch_mask 是(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵，便于之后attention focus
    # states_padding_batch 是储存padding的tensor,(batch_size, 11, 121)
    # states_hidden_BP_batch 是储存mask的tensor, (batch_size, 11, 121)
    # num_agents_accum 累计的agents的num，(1+batch_size, ) 累计的agents数量
    # agent_ids_batch 是agents id的tensor, (batch_size, 11), 如果不足6个agents，用-1补全了
    if torch.isnan(states_batch).any() or torch.isinf(states_batch).any():
        print("Warning: NaN or Inf found in states_batch.")
    return (states_batch, agents_batch_mask, states_padding_batch, states_hidden_BP_batch, num_agents_accum, agent_ids_batch)
 
 

def create_batch_C(batch): #MR+MP

    #tensor序列长度 11*121*3 #补齐T维度 到121
    sample_freq = 5 # 帧数
    sequence_length = int(24*25/sample_freq + 1) # 24s 对应的序列长度
    list_24s = [i*(1/sample_freq) for i in range(0, sequence_length)] # sequence_length 对应的game clock
    list_24s.reverse() # 反转，从24s 到0s 的变化

    #设置时间步长为序列长度，并初始化一个空的NumPy数组states_batch，用于存储状态信息，其形状将会是[batch_size, time_steps, 4]
    time_steps = sequence_length # 121
    states_batch = np.array([]).reshape(-1,time_steps,3)

    # 初始化两个空的NumPy数组，states_padding_batch和states_hidden_BP_batch，用于存储填充状态和隐藏状态
    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([]) #每一个poss的agent数量
    #agent_ids_batch = np.array([]) #每一个batch的agent ids, = batch size * 11

    #player_newids = pd.read_csv('players.csv')

    for scene in batch:

        scene_tensor = None
        #agent_ids = np.array([])
        for i in range(len(scene.agents)): # 共有11个agents
            agent = scene.agents[i]
            #agent_id = get_newid(agent.playerid,player_newids)
            #agent_ids = np.append(agent_ids,agent_id)
            #x = np.array(agent.x)[::sample_freq] #间隔sample_freq个取值
            #y = np.array(agent.y)[::sample_freq] #间隔sample_freq个取值
            #v = np.array(agent.v)[::sample_freq] #间隔sample_freq个取值
            #single_agent_tensor = torch.Tensor([[x,y,v]])
            agent_team = agent.teamid
            #t = 0 # 0:ball; 1:off; 2:def
            #if agent_team == scene.off_teamid:
            #    t = 1
            #elif agent_team == scene.def_teamid:
            #    t = 2
            if agent_team != scene.def_teamid:
                single_agent_array = np.array([[x,y,v] for x,y,v in zip(agent.x,agent.y, agent.v)][::sample_freq]) # 按照sample_freq，即5帧/s取样
                single_agent_tensor = torch.Tensor(torch.tensor([single_agent_array.transpose()]))
                single_agent_tensor = single_agent_tensor.to(torch.float)
                # (time_steps, 4) 转置-> (4, time_steps); 4: x,y,v速率,t(ball/off/def)

                if scene_tensor == None:
                    scene_tensor = single_agent_tensor
                else:
                    scene_tensor = torch.cat([scene_tensor, single_agent_tensor], dim=0) #获得6个agent的tensor shape=[6,3,time_steps]

        scene_tensor = scene_tensor
        #补齐A的维度为6
        A,D,T = scene_tensor.size()
        # 如果agents 不足11个：用-1 补齐scence_tensor,agent_ids也用-1补齐
        if A < 6:
            new_dim = torch.full((6-A, D, T), -1)
            scene_tensor = torch.cat([new_dim, scene_tensor], dim=0)
            #agent_ids = np.pad(agent_ids, (11 - A, 0), 'constant', constant_values=(-1,))

        #agent_ids_batch = np.append(agent_ids_batch,agent_ids)
        scene_tensor = torch.transpose(scene_tensor , dim0=1, dim1=2) #shape[A,T,D]
        time_24s = scene.time_24s[::sample_freq] #间隔sample_freq个取值

        # 计算头部和尾部的填充padding大小
        head_padding_size = 0
        end_padding_size =  121 - (head_padding_size+len(time_24s)+1)  #endtime～0

        #[11,121,4] 用0来补齐 2023.04.01修改 #0414update, 顶端对齐
        # 对scence_tensor 的时间T维度用0进行padding
        # TODO：为什么要做前面padding一下即有head_padding_size+1，而不是head_padding_size
        states_feat = torch.nn.functional.pad(scene_tensor,  (0, 0, head_padding_size+1, end_padding_size, 0, 0), mode='constant', value=0)
        # states_padding: padding 的位置的tensor
        #bool型，(11,121) True为padding
        states_padding = states_feat[:,:,0]
        states_padding = states_padding == 0

        #
        #mask process #according Velocity masking
        states_hidden_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_) #True为hidden
        current_step = len(time_24s) #
       # states_hidden_BP[:,:current_step] = False #实际轨迹为False

        flag = random.choice(['A','B'])

        if flag == 'A':
            split_time = len(time_24s) * 2 // 3
            states_hidden_BP[:,:split_time] = False #实际轨迹为False
        else:
            states_hidden_BP[:,:current_step+1] = False
            v_tensor = scene_tensor[:,:,2] >= 4  #获得速度张量 [A,T,1]
            index_v = split_list_by_v(v_tensor)
            for index,lst in enumerate(index_v):
                if len(lst) >= 2:
                    random_elements = random.sample(lst,2)
                    for element in random_elements:
                        s,e = element
                        states_hidden_BP[index,s:e] = True
                else:    
                    try:
                        s,e = random.choice(lst)
                    except:
                        s,e = 0,0
                        states_hidden_BP[index,s:e] = True
            states_hidden_BP[:,1:4] = False #防止出现整段全部hidden的情况

        # print(states_hidden_BP.shape)

        num_agents = np.append(num_agents, len(states_feat)) # numpy array(batch_size,) [6，6，6，6，6，6]

        # 把每个batch的states_feat, states_padding, states_hidden_BP 合并成states_batch, states_padding_batch, states_hidden_BP_batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden_BP), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64) # 现在index=0处插入0，再进行累计求和 [0, 11, 22, ....]
    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) #创建一个全为1的[A*batch,A*batch]矩阵

    for i in range(len(num_agents)):
        # 构造(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    agent_ids_batch = torch.empty(1,1)
    # states_batch是(batch_size,11,121,4)的tensor，用于储存6个agents 的轨迹，末尾被补全,用于符合模型输入
    # agents_batch_mask 是(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵，便于之后attention focus
    # states_padding_batch 是储存padding的tensor,(batch_size, 11, 121)
    # states_hidden_BP_batch 是储存mask的tensor, (batch_size, 11, 121)
    # num_agents_accum 累计的agents的num，(1+batch_size, ) 累计的agents数量
    # agent_ids_batch 是agents id的tensor, (batch_size, 11), 如果不足6个agents，用-1补全了
    return (states_batch, agents_batch_mask, states_padding_batch, states_hidden_BP_batch, num_agents_accum, agent_ids_batch)

   
  

def get_newid(playerid,df):
    if playerid in df['playerid'].values:
        newid = df[df['playerid'] == playerid]['UnifiedPlayerID'].values[0]
    else:
        newid = df['UnifiedPlayerID'].max() + 1
        df.loc[len(df)] = {'playerid': playerid, 'UnifiedPlayerID': newid}
    
    return newid
    
def create_batch_b2v(batch): #单向轨迹重建

    #tensor序列长度 11*121*3 #补齐T维度 到121
    sample_freq = 5 # 帧数
    sequence_length = int(24*25/sample_freq + 1) # 24s 对应的序列长度
    list_24s = [i*(1/sample_freq) for i in range(0, sequence_length)] # sequence_length 对应的game clock
    list_24s.reverse() # 反转，从24s 到0s 的变化

    #设置时间步长为序列长度，并初始化一个空的NumPy数组states_batch，用于存储状态信息，其形状将会是[batch_size, time_steps, 4]
    time_steps = sequence_length # 121
    states_batch = np.array([]).reshape(-1,time_steps,3)

    # 初始化两个空的NumPy数组，states_padding_batch和states_hidden_BP_batch，用于存储填充状态和隐藏状态
    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([]) #每一个poss的agent数量
    agent_ids_batch = np.array([]) #每一个batch的agent ids, = batch size * 11

    player_newids = pd.read_csv('/./mnt/nvme_share/srt02/SceneTransformer/DataLoader/players.csv')

    for scene in batch:

        scene_tensor = None
        agent_ids = np.array([])
        for i in range(len(scene.agents)): # 共有11个agents
            agent = scene.agents[i]
            #x = np.array(agent.x)[::sample_freq] #间隔sample_freq个取值
            #y = np.array(agent.y)[::sample_freq] #间隔sample_freq个取值
            #v = np.array(agent.v)[::sample_freq] #间隔sample_freq个取值
            #single_agent_tensor = torch.Tensor([[x,y,v]])
            agent_team = agent.teamid
            if agent_team != scene.def_teamid:
                single_agent_array = np.array([[x,y,v] for x,y,v in zip(agent.x,agent.y, agent.v)][::sample_freq]) # 按照sample_freq，即5帧/s取样
                single_agent_tensor = torch.Tensor(torch.tensor([single_agent_array.transpose()]))
                single_agent_tensor = single_agent_tensor.to(torch.float)
                # (time_steps, 4) 转置-> (4, time_steps); 4: x,y,v速率,t(ball/off/def)
                agent_id = get_newid(agent.playerid,player_newids)
                agent_ids = np.append(agent_ids,agent_id)

                if scene_tensor == None:
                    scene_tensor = single_agent_tensor
                else:
                    scene_tensor = torch.cat([scene_tensor, single_agent_tensor], dim=0) #获得6个agent的tensor shape=[6,3,time_steps]

        scene_tensor = scene_tensor
        #补齐A的维度为6
        A,D,T = scene_tensor.size()
        # 如果agents 不足11个：用-1 补齐scence_tensor,agent_ids也用-1补齐
        if A < 6:
            new_dim = torch.full((6-A, D, T), -1)
            scene_tensor = torch.cat([new_dim, scene_tensor], dim=0)
            agent_ids = np.pad(agent_ids, (6 - A, 0), 'constant', constant_values=(-1,))

        agent_ids_batch = np.append(agent_ids_batch,agent_ids)
        scene_tensor = torch.transpose(scene_tensor , dim0=1, dim1=2) #shape[A,T,D]
        time_24s = scene.time_24s[::sample_freq] #间隔sample_freq个取值

        # 计算头部和尾部的填充padding大小
        head_padding_size = 0
        end_padding_size =  121 - (head_padding_size+len(time_24s)+1)  #endtime～0

        #[11,121,4] 用0来补齐 2023.04.01修改 #0414update, 顶端对齐
        # 对scence_tensor 的时间T维度用0进行padding
        # TODO：为什么要做前面padding一下即有head_padding_size+1，而不是head_padding_size
        states_feat = torch.nn.functional.pad(scene_tensor,  (0, 0, head_padding_size+1, end_padding_size, 0, 0), mode='constant', value=0)
        # states_padding: padding 的位置的tensor
        #bool型，(11,121) True为padding
        states_padding = states_feat[:,:,0]
        states_padding = states_padding == 0

        #
        #mask process #according Velocity masking
        states_hidden_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_) #True为hidden
        current_step = len(time_24s) #
       # states_hidden_BP[:,:current_step] = False #实际轨迹为False

        flag = random.choice(['A','B'])

        if flag == 'A':
            split_time = len(time_24s) * 2 // 3
            states_hidden_BP[:,:split_time] = False #实际轨迹为False
        else:
            states_hidden_BP[:,:current_step+1] = False
            v_tensor = scene_tensor[:,:,2] >= 4  #获得速度张量 [A,T,1]
            index_v = split_list_by_v(v_tensor)
            for index,lst in enumerate(index_v):
                if len(lst) >= 2:
                    random_elements = random.sample(lst,2)
                    for element in random_elements:
                        s,e = element
                        states_hidden_BP[index,s:e] = True
                else:
                    try:
                        s,e = random.choice(lst)
                    except:
                        s,e = 0,0
                        states_hidden_BP[index,s:e] = True
            states_hidden_BP[:,1:4] = False #防止出现整段全部hidden的情况

        # print(states_hidden_BP.shape)

        num_agents = np.append(num_agents, len(states_feat)) # numpy array(batch_size,) [6，6，6，6，6，6]

        # 把每个batch的states_feat, states_padding, states_hidden_BP 合并成states_batch, states_padding_batch, states_hidden_BP_batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden_BP), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64) # 现在index=0处插入0，再进行累计求和 [0, 11, 22, ....]
    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) #创建一个全为1的[A*batch,A*batch]矩阵

    for i in range(len(num_agents)):
        # 构造(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    agent_ids_batch = torch.FloatTensor(agent_ids_batch)
    # states_batch是(batch_size,11,121,4)的tensor，用于储存6个agents 的轨迹，末尾被补全,用于符合模型输入
    # agents_batch_mask 是(11*batch_size，11*batch_size)矩阵，为batch_size 个分块全1矩阵，便于之后attention focus
    # states_padding_batch 是储存padding的tensor,(batch_size, 11, 121)
    # states_hidden_BP_batch 是储存mask的tensor, (batch_size, 11, 121)
    # num_agents_accum 累计的agents的num，(1+batch_size, ) 累计的agents数量
    # agent_ids_batch 是agents id的tensor, (batch_size, 11), 如果不足6个agents，用-1补全了
    return (states_batch, agents_batch_mask, states_padding_batch, states_hidden_BP_batch, num_agents_accum, agent_ids_batch)



