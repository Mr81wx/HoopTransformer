# -*- coding: utf-8 -*-



import math
import numpy as np

#计算速度 from geom import veloctity
def velocity(traj): #获得速度
    dif = traj[1:] - traj[:-1]
    v = np.sqrt((dif ** 2).sum(1)) /0.04 # feet / second
    return v


#把所有坐标转化到左边半场
def reversal_coord(court_side,coord): 
    x = coord[0]
    y = coord[1]
    if court_side:
        x = 94 - x
        y = 50 - y
    """if x < 0:
       x = -1
    if y > 50:
        y = -1
    if y < 0 :
        y = -1"""
    return([x,y])

def split_list_by_v(tensor):
    fragment_per_agent = []
    for agnet_v in tensor:
        fragments = []
        start_index = None

        for i, num in enumerate(agnet_v):
            if num == 1:
                if start_index is None:
                    start_index = i
            elif start_index is not None:
                fragments.append((start_index,i-1))
                start_index = None

        if start_index is not None:
            fragments.append((start_index,len(agnet_v)-1))
        fragment_per_agent.append(fragments)
        
    return(fragment_per_agent)