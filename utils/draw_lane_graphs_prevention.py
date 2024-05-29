import cv2
from os.path import join
from tqdm import tqdm
from math import floor
import numpy as np
from itertools import islice
from intention_prediction.utils.draw_graph import draw_g
from intention_prediction.models.map_net import map_net
import torch
import debugpy
# debugpy.listen(("localhost", 5678))
import logging
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
log = logging.getLogger("draw_module")
def main():
    # cfg = logging.getLogger("")
    r=2
    d=2
    r_d = "/r{}/d{}".format(r,d)
    time_stamp = (30,30)

    path_to_video_in = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/new_src" +r_d +  "/video_camera1.mp4".format(r,d)
    path_to_video_out_lanes = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/new_src"+r_d+"/r{}_d{}_lanes.mp4".format(r,d)
    path_to_video_out_traje = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/new_src"+r_d+"/r{}_d{}_traje.mp4".format(r,d)
    path_to_label_root = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/new_src"+r_d

    path_to_calib_root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/new_src"+r_d

    with open( join(path_to_calib_root , "velodyne_extrinsic_calibration.dat"),"r") as f:
        parameters = f.readline()
        parameters = parameters.rsplit()
        parameters = [float(x) for x in parameters]
        param_1 = parameters[3]
        param_2 = parameters[7]
        param_3 = parameters[11]
        lidar_extrinsic = np.array([[1 , 0 ,0 , param_1] , [0 , 1 ,0 , param_2] , [0 , 0 ,1 , param_3] , [0, 0 , 0 , 1]])
        lidar_extrinsic_inv = np.linalg.inv(lidar_extrinsic)

    with open(join(path_to_label_root , "processed_data", "detection_camera1" , "lanes.txt"),"r") as f:
        lanes = f.readlines()
    
    with open(join(path_to_label_root , "processed_data" ,"detection_camera1" , "trajectories.txt") , "r") as f2:
        traj = f2.readlines()

    with open(join(path_to_calib_root, "camera1_extrinsic_calibration.dat") , "r") as f3:
        rotation = f3.readline()
        rotation = rotation.rsplit()[1:]
        rotation = np.array([float(x) for x in rotation])
        
        translation = f3.readline()
        translation = translation.rsplit()[1:]
        translation = np.array([float(x) for x in translation])
        
        rotation_matrix = np.array([ [rotation[0] , rotation[1] , rotation[2] ] , 
                                    [rotation[3] , rotation[4] , rotation[5]] ,
                                    [rotation[6] , rotation[7] , rotation[8]] ] )
        
        translation_vector = np.array([ [translation[0] ,translation[1] , translation[2]]])
        
        extrinsic_matrix_extended = np.array([[ rotation[0] , rotation[1] ,rotation[2]  , translation[0] ],
                                        [ rotation[3] , rotation[4] ,rotation[5]  , translation[1] ],
                                        [ rotation[6] , rotation[7] ,rotation[8]  , translation[2] ],
                                        [0 , 0 , 0 , 1]])
        
        extrinsic_matrix  = np.array([[ rotation[0] , rotation[1] ,rotation[2]  , translation[0] ],
                                        [ rotation[3] , rotation[4] ,rotation[5]  , translation[1] ],
                                        [ rotation[6] , rotation[7] ,rotation[8]  , translation[2] ],
                                         ])

    cap_reader = cv2.VideoCapture(path_to_video_in )
    total_frames=cap_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=cap_reader.get(cv2.CAP_PROP_FPS)
    H,W = int(cap_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_begin = (time_stamp[0]*60 + time_stamp[1])*fps
    cap_reader.set(cv2.CAP_PROP_POS_FRAMES , frame_begin)

    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 10000)
    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 8900)
    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 7500)

    with open(join(path_to_calib_root , "camera1_intrinsic_calibration.dat") , "r") as f3:
            _ = f3.readline()
            intrinsic_params= f3.readline()
            intrinsic_params = intrinsic_params.rsplit()[1:]
            fx=float(intrinsic_params[0])
            fy=float(intrinsic_params[3])
            ox=float(intrinsic_params[2])
            oy=float(intrinsic_params[4])
            print("Intrinsic camera parameters are fx= {} fy={} ox={} oy={}".format(fx,fy,ox,oy))
            intrinsic_params = np.array([float(x) for x in intrinsic_params])

            intrinsic_matrix = np.array([ [intrinsic_params[0] , intrinsic_params[1] , intrinsic_params[2]],
                                            [0, intrinsic_params[3] , intrinsic_params[4]],
                                             [0,0,1] ])
            
            intrinsic_matrix_extended = np.array([ [intrinsic_params[0] , intrinsic_params[1] , intrinsic_params[2], 0],
                                            [0, intrinsic_params[3] , intrinsic_params[4], 0],
                                            [0,0,1 , 0]  ] , 
                                        )

            

            distortion_params = f3.readline()
            distortion_params = distortion_params.rsplit()[1:]
            distortion_params = [float(x) for x in distortion_params]
            k1=distortion_params[0]
            k2=distortion_params[1]
            k3=distortion_params[2]
            p1=distortion_params[3]
            p2=distortion_params[4]


    input("Estimated Optical Centre vs Camera center coordintates {} {} {} {}".format(ox,oy,H/2,W/2))
    if ox-W/2!=0: o_diff_x = ox-W/2
    
    if oy-H/2!=0: o_diff_y = oy-H/2

    cap_writer_lanes = cv2.VideoWriter(path_to_video_out_lanes , cv2.VideoWriter_fourcc(*"mp4v") , fps , (W,H))

    colour_map = {0: (0,0,255) , 1:(0,255,0),
                  2:(255,255,0), 3:(0,255,255) , 
                  4:(255,0,255)}
    with tqdm(total=total_frames) as pbar:
        while True:
            try:
                ret,val=cap_reader.read()
                pbar.update(1)
                current_frame = cap_reader.get(cv2.CAP_PROP_POS_FRAMES)
                pbar.set_description_str("Now process frame no.{:0.2f}/{:0.1f}".format(current_frame,total_frames))
                if ret:
                    frame = val
                    frame_idx = int(current_frame)
                    pts=[]

                    polyline_list = []  #list for holding points on polylines on current frame

                    for idx, lane_info in enumerate(lanes):
                        lanes_processed = lane_info.rsplit()  #lane polyline coeeffs
                        lanes_processed = [float(x) for x in lanes_processed]

                        if lanes_processed[0] == frame_idx and int(lanes_processed[6]) :  #6=isDetected
                            
                                lane_id = int(lanes_processed[1])  #id of lane (1-4) lanes total in highway
                                colour = colour_map[lane_id]
                                c0,c1,c2,c3=tuple(lanes_processed[2:6]) #coeffs
                                polyline_list.append(np.array([c0,c1,c2,c3]))

                                print("Parameter list for lane {} is {}:".format(idx,lanes_processed[2:6])) 
                                ky_l=[]
                                for kx in iter(np.linspace(start=10,stop=50 , num=3000)):
                                    
                                    ky = c3*kx**3 + c2*kx**2 + c1*kx + c0 

                                    vehicle_homogeneous = np.array([[kx],[ky],[-2],[1]])

                                    real_world_vect_homogeneous =  lidar_extrinsic_inv @ vehicle_homogeneous #convert to vehicle system

                                    # real_world_vect_homogeneous[0] , real_world_vect_homogeneous[1] , real_world_vect_homogeneous[2] = \
                                    #     -real_world_vect_homogeneous[1], -real_world_vect_homogeneous[2], real_world_vect_homogeneous[0]

                                    camera_vector = extrinsic_matrix_extended @ real_world_vect_homogeneous 

                                    # camera_vector[1] = -camera_vector[0]
                                    # camera_vector[0] = -camera_vector[1]
                                    # camera_vector[2] = -camera_vector[2]
                                    
                                    frame_vector =  intrinsic_matrix_extended @ camera_vector#apply extrinsic and intrinsic

                                    x_tilde, y_tilde, z_tilde = frame_vector[0] , frame_vector[1] , frame_vector[2] 

                                    x = x_tilde/z_tilde #convert to non homogeneous
                                    y = y_tilde/z_tilde

                                    #Undistortion####
                                    x=x-(ox-W/2) #remove difference of bad calibrated parameter
                                    y=y-(oy-H/2)
                                    
                                    x = (x-W/2)/fx #cetner,norm
                                    y = (y-H/2)/fy

                                    r=np.sqrt(x**2+y**2)
                                    x = (x)*(1 + k1*r**2 + k2*r**4 + k3*r**6)+ (2*p1*x*y+p2*(r**2+2*x**2)) 
                                    y = (y) * (1 + k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 +2*y**2) + 2*p2*x*y
                                     
                                    x = fx*x + W/2 #uncenter,unnorm
                                    y = fy*y + H/2
                                    ###################

                                    pts.append([y,x,colour])

                                    #immage annot
                                    org = (50, 50)                              
                                    font = cv2.FONT_HERSHEY_SIMPLEX 
                                    fontScale = 1
                                    color=colour
                                    thickness = 1
                                    frame = cv2.putText(frame, "{}".format(lanes_processed[2]), (50,lane_id*25 + 50), font,  fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "{}".format(lanes_processed[3]), (300,lane_id*25+ 50), font,  fontScale, color, thickness, cv2.LINE_AA) 
                                    # frame = cv2.putText(frame, "{}".format(lanes_processed[4]), (600,50), font,  
                                    #                 fontScale, color, thickness, cv2.LINE_AA) 
                                    # frame = cv2.putText(frame, "{}".format(lanes_processed[5]), (900,50), font,  
                                    #                 fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "0:blue", (50,700), font,  fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "1:green", (50+50,700), font,  fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "2:yellow", (50+100,700), font,   fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "3:cyan", (50+150,700), font,  fontScale, color, thickness, cv2.LINE_AA) 
                                    frame = cv2.putText(frame, "4:magenta", (50+200,700), font,  
                                                    fontScale, color, thickness, cv2.LINE_AA) 
                    c=0
                    for points in pts:
                        
                        x = points[1]
                        y = points[0]
                        color=points[2]
                        if x>0 and x<W and y>0 and y<H:
                            c+=1
                            frame[int(y[0]),int(x[0]),:] = color  #draw green line at x,y

                    G=nx.Graph()
                    
                    for idx, points_discreet in enumerate(islice(pts , 0 , len(pts) , 25)):
                        x = points_discreet[1]
                        y = points_discreet[0]
                        x = x[0]
                        y = y[0]
                        pyData = Data(x = torch.tensor([x , y]))
                        G.add_node(idx , loc = (x,y))
                        
                        if idx>0:
                            G.add_edge(idx,idx-1)

                    # nx.draw(G)
                    # draw_g(G , frame)

                    # for i in range(1 , pyData.num_nodes ):
                    #     pyData.update()

                    if polyline_list!=[]:
                        frame_map = map_net(lane_coefficient= polyline_list,
                                            discretization_res = (3, 3))
                        frame_graph = frame_map.get_graph()
                        print(frame_graph.nodes.data())
                        draw_g(frame_graph , frame)
                    
                    print("Lane Points drawn on image {}".format(c))
                    # distortion_params=np.array(distortion_params)
                    # newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_params, (W,H), 0, (W,H))
                    # frame = cv2.undistort(frame, intrinsic_matrix, distortion_params, None, newcamera_mtx)

                
                    cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/utils/test_video_frame.png",frame)
                    # input("Waiting")
                    cap_writer_lanes.write(frame)

                else:
                    break
            except KeyboardInterrupt as e:
                cap_reader.release()
                cap_writer_lanes.release()

if __name__=="__main__":

    main()