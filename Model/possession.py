from .geometry import velocity,reversal_coord,split_list_by_v
import numpy as np

#提取进攻方球员
class Player():
    def __init__(self,playerid,teamid):
        self.playerid = playerid
        self.teamid = teamid
        self.x = []
        self.y = []
        self.timeframe = []
        self.v = []
        self.valid = []
        


class Possession():
    def __init__(self,off_team_id,def_team_id,event,gameid):
        self.gameid = gameid
        self.id = event['eventId']
        self.moments = event['moments']
        self.off_side = event['Offense_side'] 
        self.off_teamid = off_team_id
        self.def_teamid = def_team_id
        self.timeframe, self.time_game,self.time_24s = self.get_time()
        self.agents = self.get_agents_list()
        self.play_label = ""
        self.action_label = ""
        self.outcome = ""
    
    def get_time(self):
        time_list = []
        time24_list = []
        time_gamelist = []
        for i in range(0,len(self.moments)):
            moment = self.moments[i] 
            time_list.append(moment[1])
            time_gamelist.append(moment[2])
            time24_list.append(moment[3])
        
        return time_list, time_gamelist, time24_list


    def get_agents_list(self):
        dat = self.moments[0][5]
        #player_teamid_list = [sublist[0] for sublist in dat] #取所有agent的teamid
        #ball = Player(-1,-1)
        agents = []
        #indices = [index for index, element in enumerate(player_teamid_list) if element == self.off_teamid or self.off_teamid ] #得到进攻方的index
        for agent in dat:
            agents.append(Player(agent[1],agent[0]))
        
        agents = self.get_agents_feature(agents)
        return agents

    def get_agents_feature(self,agents_list):
        court_side = self.moments[0][5][0][2] > 47.0 #判断是在左边半场还是右边半场,右为1，需要转换
        for player in agents_list: 
            player.timeframe = self.timeframe
            #get x，y
            for i, moment in enumerate(self.moments):  
                valid = 0       
                for item in moment[5]: #11个item里查找
                    if player.playerid == item[1]:
                        coord = reversal_coord(court_side,[item[2],item[3]])
                        player.x.append(coord[0])
                        player.y.append(coord[1])
                        valid = 1
                        player.valid.append(valid)
                        continue
                if valid == 0: #如果找不到，则invalid
                    player.x.append(-1)
                    player.y.append(-1)
                    player.valid.append(0)
            
            #get v
            coords = np.stack((player.x, player.y), axis=1)
            if len(coords)<2: #如果轨迹长度太短，直接跳出
                continue
            else: 
                v = velocity(coords)
                player.v = np.append(v,v[-1]) #计算每个时间点的速度
        
        return agents_list
    
        

    
def Convert_scene_tra(data): 
    game_id = data['gameid']
    hometeam_id = data['events'][0]['home']['teamid']
    awayteam_id = data['events'][0]['visitor']['teamid']
    scene_list = []
    for index,possession in enumerate(data['events']):  
        if possession.__contains__('Offense_side'):
            off_side = possession['Offense_side']
            off_team_id = (hometeam_id if off_side == "H" else awayteam_id)
            def_team_id = (awayteam_id if off_side == "H" else hometeam_id)
            moments = possession['moments']
            if len(moments) >= 50:
                scene = Possession(off_team_id,def_team_id,possession,game_id)
                scene_list.append(scene)
    
    return(scene_list)