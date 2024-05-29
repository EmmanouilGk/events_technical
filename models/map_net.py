from itertools import islice
from torch.nn import Module
from typing import List

from torch_geometric.data import Data
from numpy import linspace
import numpy as np
from numpy import absolute as np_abs

from numpy.linalg import norm
import debugpy

from scipy.spatial.distance import euclidean

import networkx as nx

from ..utils.projective_trans import coordinate_transforms

"""
MAP NET 

"""


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
        self.G = nx.Graph()
        self.cT = coordinate_transforms()
        
        #distritize lanes
        middle_x_list, middle_y_list , self.id_list , self.node_val = self.distritize_polyline(self.lane_coefficient)

        self.middle_x_list = list(middle_x_list)
        self.middle_y_list = list(middle_y_list)
        self.id_list = list(self.id_list)

        self.list_edges = self.construct_adjacency_successor()

    def get_graph(self):
        return self.G

    def distritize_polyline(self , polylines_list:List):
        """
        Positions must be given in ordered fashion ie increasing lateral distance or reverse
        """
        node_pos=[]
        node_val = 0
        for idx_lane,polylines in enumerate(polylines_list):
            lateral_equation = lambda x : polylines[0] + polylines[1]*x + polylines[2]*x**2
        
            start,end = 0 , 100
            # num_int = int(total_length/(norm(self.discretization_res)))
            num_int = 10
            total_length = end

            for longitudinal_offset in linspace(start = start , stop= end,  num=num_int ):
                node_val += 1 
                middle_x = start+longitudinal_offset
                middle_y = lateral_equation(x=middle_x)

                middle_x,middle_y = self.cT.perspective(middle_x , middle_y)
                if middle_y<0 or middle_y>1800:continue
                node_pos.append((middle_x , middle_y , idx_lane , node_val))

        
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
        list_edge_coo = []
        frame_points = zip(self.middle_x_list , self.middle_y_list, self.id_list , self.node_val)
        
        for idx , center_point_annotation in enumerate(zip(self.middle_x_list , self.middle_y_list, self.id_list , self.node_val)):

            xy = (center_point_annotation[0][0],center_point_annotation[1][0])

            center_point=list(xy) #2d coord in list type for euclid calc

            id = center_point_annotation[2]
            node_id = center_point_annotation[3]
            self.G.add_node(node_id , loc = xy) #add first node 

            #compute distances to id closest neighbour

            points_same_lane = list(filter(lambda y: y[2] == id , list(zip(self.middle_x_list , self.middle_y_list, self.id_list ,self.node_val)) )) #get all center points on same polyline y

            distances_same_lane = []
            for x,y,i,n in points_same_lane:
                x=x[0]
                y=y[0]
                euclid = euclidean(center_point , [x, y])
                distances_same_lane.append(euclid)
            distances_same_lane = sorted(distances_same_lane)
            # distances_same_lane = sorted([euclidean(center_point , [y[0][0][0] ,y[1][0][0]]) for y in points_same_lane])  #get distance of current point to all points on same line (y_coordinate)
            
            distances_same_lane = list(filter(lambda x : int(x)>0 , distances_same_lane))

            print("Disatnces same lane {}".format(distances_same_lane))

            for center_point_neighbour in frame_points: 
                center_point_succ = [center_point_neighbour[0][0] , center_point_neighbour[1][0]]  #candidate neighbour xy
                id_succ = center_point_neighbour[2] 
                node_id_succ = center_point_neighbour[3]                                          #candiate neibhbour id
                dist=euclidean(center_point_succ,center_point)
                print("Dist to succ {} and succ id {} and id {}".format(dist,id_succ , id))
                if dist == distances_same_lane[0] and int(dist)!=0 and id==id_succ and node_id_succ != node_id:       #euclid dist neibour,current center point
                    successor_point = tuple(center_point_succ)
                    id_successor  = id_succ
                    #nodes and edges [neighour]
                    self.G.add_node(node_id_succ , loc=successor_point) #add closest point
                    list_edge_coo.append((node_id,node_id_succ))
                    self.G.add_edge(node_id,node_id_succ)


        print(list_edge_coo)
        print(self.G)
        return list_edge_coo

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
            list_edge_coo.append(())
            G.add_edge(id,id_successor)


        return G , list_edge_coo