import numpy as np
from os.path import join
import cv2
class coordinate_transforms():
        def __init__(self) -> None:
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
                    self.lidar_extrinsic = np.array([[1 , 0 ,0 , param_1] , [0 , 1 ,0 , param_2] , [0 , 0 ,1 , param_3] , [0, 0 , 0 , 1]])
                    self.lidar_extrinsic_inv = np.linalg.inv(self.lidar_extrinsic)

                with open(join(path_to_label_root , "processed_data", "detection_camera1" , "lanes.txt"),"r") as f:
                    self.lanes = f.readlines()
                
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
                    
                    self.extrinsic_matrix_extended = np.array([[ rotation[0] , rotation[1] ,rotation[2]  , translation[0] ],
                                                    [ rotation[3] , rotation[4] ,rotation[5]  , translation[1] ],
                                                    [ rotation[6] , rotation[7] ,rotation[8]  , translation[2] ],
                                                    [0 , 0 , 0 , 1]])
                    
                    self.extrinsic_matrix  = np.array([[ rotation[0] , rotation[1] ,rotation[2]  , translation[0] ],
                                                    [ rotation[3] , rotation[4] ,rotation[5]  , translation[1] ],
                                                    [ rotation[6] , rotation[7] ,rotation[8]  , translation[2] ],
                                                    ])

                    cap_reader = cv2.VideoCapture(path_to_video_in )
                    self.total_frames=cap_reader.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps=cap_reader.get(cv2.CAP_PROP_FPS)
                    self.H,self.W = int(cap_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_begin = (time_stamp[0]*60 + time_stamp[1])*fps
                    cap_reader.set(cv2.CAP_PROP_POS_FRAMES , frame_begin)

                    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 10000)
                    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 8900)
                    # cap_reader.set(cv2.CAP_PROP_POS_FRAMES , 7500)

                with open(join(path_to_calib_root , "camera1_intrinsic_calibration.dat") , "r") as f3:
                        _ = f3.readline()
                        intrinsic_params= f3.readline()
                        intrinsic_params = intrinsic_params.rsplit()[1:]
                        self.fx=float(intrinsic_params[0])
                        self.fy=float(intrinsic_params[3])
                        self.ox=float(intrinsic_params[2])
                        self.oy=float(intrinsic_params[4])
                        print("Intrinsic camera parameters are fx= {} fy={} ox={} oy={}".format(self.fx,self.fy,self.ox,self.oy))
                        intrinsic_params = np.array([float(x) for x in intrinsic_params])

                        self.intrinsic_matrix = np.array([ [intrinsic_params[0] , intrinsic_params[1] , intrinsic_params[2]],
                                                        [0, intrinsic_params[3] , intrinsic_params[4]],
                                                        [0,0,1] ])
                        
                        self.intrinsic_matrix_extended = np.array([ [intrinsic_params[0] , intrinsic_params[1] , intrinsic_params[2], 0],
                                                        [0, intrinsic_params[3] , intrinsic_params[4], 0],
                                                        [0,0,1 , 0]  ] , 
                                                    )

                        

                        distortion_params = f3.readline()
                        distortion_params = distortion_params.rsplit()[1:]
                        distortion_params = [float(x) for x in distortion_params]
                        self.k1=distortion_params[0]
                        self.k2=distortion_params[1]
                        self.k3=distortion_params[2]
                        self.p1=distortion_params[3]
                        self.p2=distortion_params[4]


                pass

        def perspective(self, x , y):  

            kx=x
            ky=y
            vehicle_homogeneous = np.array([[kx],[ky],[-2],[1]])

            real_world_vect_homogeneous =  self.lidar_extrinsic_inv @ vehicle_homogeneous #convert to vehicle system

            # real_world_vect_homogeneous[0] , real_world_vect_homogeneous[1] , real_world_vect_homogeneous[2] = \
            #     -real_world_vect_homogeneous[1], -real_world_vect_homogeneous[2], real_world_vect_homogeneous[0]

            camera_vector = self.extrinsic_matrix_extended @ real_world_vect_homogeneous 

            # camera_vector[1] = -camera_vector[0]
            # camera_vector[0] = -camera_vector[1]
            # camera_vector[2] = -camera_vector[2]
            
            frame_vector =  self.intrinsic_matrix_extended @ camera_vector#apply extrinsic and intrinsic

            x_tilde, y_tilde, z_tilde = frame_vector[0] , frame_vector[1] , frame_vector[2] 

            x = x_tilde/z_tilde #convert to non homogeneous
            y = y_tilde/z_tilde

            #Undistortion####
            x=x-(self.ox-self.W/2) #remove difference of bad calibrated parameter
            y=y-(self.oy-self.H/2)
            
            x = (x-self.W/2)/self.fx #cetner,norm
            y = (y-self.H/2)/self.fy

            r=np.sqrt(x**2+y**2)
            x = (x)*(1 + self.k1*r**2 + self.k2*r**4 + self.k3*r**6)+ (2*self.p1*x*y+self.p2*(r**2+2*x**2)) 
            y = (y) * (1 + self.k1*r**2 + self.k2*r**4 + self.k3*r**6) + self.p1*(r**2 +2*y**2) + 2*self.p2*x*y
                
            x = self.fx*x + self.W/2 #uncenter,unnorm
            y = self.fy*y + self.H/2

            return x,y