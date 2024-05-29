from itertools import islice
from torch.nn import Module
from typing import List

from torch_geometric.data import Data
from numpy import linspace
import numpy as np
from numpy import absolute as np_abs

from numpy.linalg import norm

from scipy.spatial.distance import euclidean

import networkx as nx

class map_net(Module):
    """
    construct graph representation of (1 frame) 
    1 sec = fps graphs .
    Input:
        lane_coefficient: lane parameters as polylines [c0 , c1 , c2]
        discretization_res : step size in road mm for discretization (tuple dx,dy)
        TODO: streaming data read (??)
        TODO: consider const len discretiazed dsemgent by first generating polyline of thosuands of points in x,y and then discretitizing. ->more time consuming ? unless generator-like
    """
    def __init__(self, 
                 lane_coefficient,
                 discretization_res
                 ):
        
        super(map_net , ).__init__()
        self.lane_coefficient= lane_coefficient


        self.discretization_res = list(discretization_res)
        
        #distritize lanes
        middle_x_list, middle_y_list , self.id_list = self.distritize_polyline(self.lane_coefficient)
        self.middle_x_list = list(middle_x_list)
        self.middle_y_list = list(middle_y_list)
        self.id_list = list(self.id_list)

        G , list_edges = self.construct_adjacency_successor()

    def distritize_polyline(self , polylines_list:List):
        """
        Positions must be given in ordered fashion ie increasing lateral distance or reverse
        """
        node_pos=[]
        print(len(polylines_list))
        for idx,polylines in enumerate(polylines_list):
            lateral_equation = lambda x : polylines[0] + polylines[1]*x + polylines[2]*x**2
        
            start,end = 0 , 100
            total_length = end

            print(self.discretization_res)

            for idx in linspace(start = start , stop= end,  num=int(total_length/int(norm(self.discretization_res))) ):
                
                middle_x = ((start+idx)+ self.discretization_res[0])/2
                middle_y = lateral_equation(x=middle_x)
                node_pos.append((middle_x , middle_y , idx))

        print(node_pos)
        
        return tuple(zip(*node_pos)) #returns list of middle_x, list middle_y ,idx polyline
    
    def construct_adjacency_successor(self):
        """
        construct adjacency graph for only the successor (connectivity) lane
        Input:
          list of discretized edges for all lanes in the graph (class state)
        returns:
          graph G(v) from nx
          list of edges in graph as coo format for pyG for only succesor lane nodes

        """
        G = nx.Graph()
        list_edge_coo = []
        frame_points = enumerate(zip(self.middle_x_list , self.middle_y_list))
        id_list= self.id_list
        for idx, center_point , id in zip(frame_points , id_list):

            loc = (center_point[0],center_point[1])
            lane_point_dict  = (id ,{"location" : loc})

            points_same_lane = list(filter(lambda y: y[2] == id , list(zip(self.middle_x_list , self.middle_y_list, self.id_list)) )) #get all center points on same polyline y
            
            distances_same_lane = sorted([euclidean(center_point , (y[0] ,y[1])) for y in points_same_lane])  #get distance of current point to all points on same line (y_coordinate)
            
            for center_points_succ,id_succ in zip(frame_points , id_list): 
                if euclidean(center_points_succ,center_point) == distances_same_lane[0] : 
                    successor_point = center_points_succ
                    id_successor = id_succ

            G.add_node(id) #add first node 
            G.add_node(successor_point) #add closest point

            # G.add_node_from(lane_point_dict)         #add one node per line center_point
            list_edge_coo.append((id,id_successor))
            G.add_edge(id,id_successor)


        return G , list_edge_coo

    def construct_adjacency_neighbour(self):
        """
        construct adjacency graph for only the successor (connectivity) lane
        Input:
          list of discretized edges for all lanes in the graph (class state)
        returns:
          graph G(v) from nx
          list of edges in graph as coo format for pyG for only succesor lane nodes

        """
        G = nx.Graph()
        list_edge_coo = []
        frame_points = enumerate(zip(self.middle_x_list , self.middle_y_list))
        id_list= self.id_list
        for idx, center_point , id in zip(frame_points , id_list):

            lane_point_dict  = (id ,{"location" : loc})

            points_neighbouring_line = list(filter(lambda y: y[2] == id+1 or y[2] == id-1 , list(zip(self.middle_x_list , self.middle_y_list, self.id_list)) )) #get all center points on same polyline y
            
            distances_same_lane = sorted([euclidean(center_point , (y[0] ,y[1])) for y in points_neighbouring_line])  #get distance of current point to all points on same line (y_coordinate)
            
            for center_points_succ,id_succ in zip(frame_points , id_list): 
                if euclidean(center_points_succ,center_point) == distances_same_lane[0] : 
                    successor_point = center_points_succ
                    id_successor = id_succ

            G.add_node(id) #add first node 
            G.add_node(successor_point) #add closest point

            # G.add_node_from(lane_point_dict)         #add one node per line center_point
            list_edge_coo.append((id,id_successor))
            G.add_edge(id,id_successor)


        return G , list_edge_coo