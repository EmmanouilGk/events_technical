import os
from typing import Any, Iterable, List, Tuple
from ..src.train_utils import bisect_right

import cv2
import math
from math import floor,ceil
import sys

from numpy import warnings
import torch
from torch.utils.data import Dataset, ConcatDataset,IterableDataset
from os.path import join
from glob import glob
from torchvision.transforms import Compose , Resize,  ToTensor , CenterCrop
import fnmatch
from ..conf.conf_py import _PADDED_FRAMES
import pandas as pd
from torchvision.transforms.v2 import ToImage
import traceback
from tqdm import tqdm

def compute_weights(ds, 
                    save_path:str,
                    save = True,
                    custom_scaling:int =1,
                    ):
           """
           Compute balancing wegiths for torch.utils.BatchSampler 
           custom scaling for the minority class:extrac scaling of weights
           """
           N=len(ds)

           labels=[]
           count_lk , count_llc, count_rlc = 0, 0, 0
           for j in tqdm(range(N)):
                label=ds[j][1]
                labels.append(int(label))
                print(label)
                if label==0:count_lk+=1
                if label==1:count_llc+=1
                if label==2:count_rlc+=1
           class_counts = list((custom_scaling*count_lk , count_llc , count_rlc)) #extra scaling for minority class
           num_samples = sum(list((count_lk , custom_scaling , count_rlc)))
           class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
           weights = [class_weights[labels[i]] for i in range(int(num_samples))]
           weights= torch.DoubleTensor(weights)

           if save:
               torch.save(weights, save_path)


           weight_dict = {"LK" : weights[0] , "LLC":weights[1], "RLC":weights[2]}
           return weights , class_weights, weight_dict

def compute_weights_binary_cls(ds, 
                    save_path:str,
                    save = True,
                    custom_scaling:int =1,
                    ):
           """
           Compute balancing wegiths for torch.utils.BatchSampler 
           custom scaling for the minority class:extrac scaling of weights
           """
           N=len(ds)

           labels=[]
           count_lk , count_llc, count_rlc = 0, 0, 0
           for j in tqdm(range(N)):
                label=ds[j][1]
                labels.append(int(label))
                print(label)
                if label==2:count_lk+=1
                if label==0:count_llc+=1
                if label==1:count_rlc+=1
           class_counts = list(( count_llc , count_rlc)) #extra scaling for minority class
           num_samples = sum(list(( custom_scaling , count_rlc)))
           class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
           weights = [class_weights[labels[i]] for i in range(int(num_samples))]
           weights= torch.DoubleTensor(weights)

           if save:
               torch.save(weights, save_path)


           weight_dict = {"LLC":weights[0], "RLC":weights[1]}
           return weights , class_weights, weight_dict


def _calculate_weigths_classes(maneuvers , lk):
    lane_change=[]
    for maneuver in maneuvers:
        lane_change.append(maneuver[2])

    rlc = len(list(filter(lambda x :x == 4 , lane_change)))
    llc = len(list(filter(lambda x :x == 3 , lane_change)))
    lk  = lk
    _sum = rlc + llc +lk

    w_rlc = (_sum)/rlc
    w_llc=(_sum)/llc
    w_lk = (_sum)/lk

    print("Weights lk,llc,rlc {} {} {}".format(w_lk ,w_llc,w_rlc))
    
    return {"LLC" : w_llc,
            "RLC": w_rlc,
            "LK": w_lk}

def _segment_video(
                   **kwargs):
    
    if kwargs["data_path"]:
        for i, src in enumerate(kwargs["data_path"]):
            video = src[1]
            processed_data = src[0]
            i=i+2       
            input(processed_data)
            dstp =  os.path.abspath(join(processed_data, os.pardir))
            print(dstp)
            
            if not os.path.isdir(join(dstp,"segmented_frames")):os.mkdir(join(dstp,"segmented_frames"))

            # dstp = join(dstp,"segmented_frames")

            print(src)
            print(dstp)
            dstp = join(dstp,"segmented_frames")
            cap = cv2.VideoCapture(video)
            frame_idx=0
            with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
                pbar.set_description_str("processing video-labels {}".format(os.path.basename(video)))
                while True:
                    if cap.isOpened():
                        ret,frame = cap.read()
                        dstp_frame = join(dstp , str(frame_idx) + ".png")
                        cv2.imwrite(dstp_frame , frame)
                        frame_idx+=1
                        pbar.update(1)
                    else:
                        break

        sys.exit()

    
    cap = cv2.VideoCapture(src)
    frame_idx=0
    with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
        while True:
            if cap.isOpened():
                ret,frame = cap.read()
                dstp_frame = join(dstp , str(frame_idx) + ".png")
                cv2.imwrite(dstp_frame , frame)
                frame_idx+=1
                pbar.update(1)
            else:
                break


def _read_lane_change_labels(label_root):
    """
    read lanechanges.txt,
    return maneuver_info: List[List[int]]
    """
    maneuver_sequences = []
    id_sequences = []


    with open(label_root , "r") as labels:
        annotations = labels.readlines()
    for i,maneuver_info in enumerate(annotations):
        annotations[i] = maneuver_info[:-1] #rmv newline char
        annotations[i] = [int(x) for x in maneuver_info.rsplit(" ")] #extract info as list of list of int
        maneuver_sequences.append(annotations[i][:])  #append maneuver info
        id_sequences.append(annotations[i][0])
        
        
    return maneuver_sequences , id_sequences


def _read_detections_labels(detections_root):
    """
    read car bbox detections
    """
    detections=[]
    with open(detections_root , "r") as f:
        while True:
            line = f.readline()
            line = line[:-1]
            _line_temp=[]
            for x in line.rsplit(" "):
                if x=="":continue
                _line_temp.append(float(x)) 
            
            line = _line_temp
                
            assert type(line)==list
            if not line:
                break
            elif line[2]!=1 and line[2]!= 2:   #exclude pedestrians,cyclists
                
                detections.append((line[0] ,line[1] ,line[2], line[3:7]))    #1:frame,2:object type annotation,3:bbox x,y tuple
    
    return detections


def _read_labels_gt(labels_root):
    """
    read car bbox detections
    """
    detections=[]
    df = pd.read_csv(labels_root, sep=" ")

    with open(labels_root , "r") as f:
        while True:
            line = f.readline()
            line = line[:-1]
            line_p=[]
            for x in line.rsplit(" "):
                if x[1:-4]=="":continue

                x= int(float(x[1:-4])*10**int(x[-1]))

                line_p.append(x)
            line=line_p
            assert type(line)==list
            if not line:
                break
            else:
                detections.append(line)    #frame,id,x,y,width,height
    
    return detections

class base_dataset():

    def __init__(self,
                 root,
                 label_root) -> None:

            self.labels=_read_lane_change_labels(label_root)
            
            

class prevention_dataset_train(Dataset):
    """
    map style dataset:
    Every _PADDED_FRAMES frames -> one prediction (Lane Keep)
    Every delta frames (defined in lane_changes.txt) -> one pred (Lane change right/left)

    return 4d rgb tensor

    model always takes _PADDED frames and makes one prediction
    """

    def __init__(self,
                 root:str,
                 label_root:str,
                 **kwargs) -> None:
        super().__init__()
        self.H = 650
        self.W = 650

        self.roi_H = 200
        self.roi_W = 200

        if "detection_root" in kwargs: self.detection_root=kwargs["detection_root"]
        if "gt_root" in kwargs: self.gt_root = kwargs["gt_root"]
        if "desc" in kwargs: self.desc = kwargs["desc"]

        self.transform = Compose([ ToImage() , CenterCrop((self.H,self.W)) ]) #transfor for each read frame

        self.transform_roi = Compose([ ToImage() , CenterCrop((self.roi_H,self.roi_W)) ])

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)[0]
        self.ids_seq = _read_lane_change_labels(label_root)[1]
        self.class_map = {"LK":2, "LLC":0, "RLC":1}


        #assign lane change clases

        self.train_split = math.floor(len(self.labels)*0.8)

        
        self.labels = self.labels[:self.train_split]#train split
   
        for maneuver_info,id_car in zip(self.labels,self.ids_seq):

            lane_start = maneuver_info[3]
            if lane_start<0: continue

            lane_end = maneuver_info[4]
            maneuver_event = maneuver_info[2] #maneuver type
            id_car = abs(maneuver_info[0])
            if id_car<0:continue
            # if lane_end-lane_start>72:continue  #skip maneuvers taking too long
            if maneuver_event==4: maneuver_event="RLC"
            if maneuver_event==3: maneuver_event="LLC"
            
            assert lane_start>0 and lane_end>0, "Expected positive maneuver labels,got {} {}".format(lane_start , lane_end)

            frames_paths = sorted([join(root , str(x) + ".png") for x in range(lane_start-5 , lane_end)])

            for x in frames_paths: 
                assert os.path.isfile(x),"No file/bad file found at path {}".format(x)
            
            self.data.append([frames_paths , maneuver_event , id_car])
            

        if 0:
            #assign lane keep clases
            i=0
            lk_counter=0

            for maneuver in self.labels:
                
                frames_paths = sorted([join(root , str(x) + ".png") for x in range( i , maneuver[4])])
                for j in range(len(frames_paths)//_PADDED_FRAMES , 10):
                    counter = 0
                    _temp_list = []
                    for frame in frames_paths:
                        _temp_list.append(frame)

                        if counter==_PADDED_FRAMES:
                            self.data.append([_temp_list , "LK"])
                            lk_counter+=1
                            
                            break
                        else:
                            counter+=1
                
                i=maneuver[5] #assingn next start to end of current manuver
                if lk_counter>20: break

            # for j in range(0,20,self._MAX_VIDEO_FRAMES):
            #     frames_temp = [] 
            #     for frame_path in glob(join( root, "*.png")):
            #         frames_temp.append(frame_path)
            #         if len(frames_temp)==20:break
                            
            #     for maneuver_info in self.labels:
            #         if frame_path in range(maneuver_info[5] - maneuver_info[4]):
                
            self.weights=_calculate_weigths_classes(maneuvers=self.labels, lk=lk_counter)

        self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter=0,0,0
        for item in self.data:
            if item[1]=="LK":self.lane_keep_counter+=1
            elif item[1]=="LLC":self.lane_change_right_counter+=1
            elif item[1]=="RLC":self.lane_change_left_counter+=1
            print("Currently have {} lane keep, {} left lane change , {} right lane change".format(self.lane_keep_counter,self.lane_change_left_counter,self.lane_change_right_counter))

        
        ##filter maneuvers by existing detect
        self.gt = _read_labels_gt(labels_root= self.gt_root)
        print(self.gt)
        self.detections=_read_detections_labels(detections_root= self.detection_root)

        # input("Label annnotations are {}".format(self.gt))

        self.labels = list(filter(lambda x : self._check_exist(x[0])==True , self.labels))  #check if labelled frame is in detections txt
        # self.test_stuff()


    def _get_desc(self):
        return self.desc
    
    def _check_exist(self,frame):
        for j in self.detections:
            if j[0]==frame:return True
        
        return False
    
    def _print_stat(self):
        return "LK,LLC,RLC: {}/ {} / {}".format(self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter)

    def crop_frames(self,frames,bboxes):

        delta_y=20 #pixels left right tolerance
        delta_x = 30
        frame_cropped=[]
        
        for i,frame in enumerate(frames):
            try:
            
                if bboxes != [-1]*4:
                    frame = frame[floor(bboxes[i][2]) - delta_y :ceil(bboxes[i][3]) +delta_y+20 , floor(bboxes[i][0]) -delta_x:ceil(bboxes[i][1])+delta_x ] #crop
         
                elif bboxes == [-1]*4:
                    frame = frame

                assert frame.shape[0]> abs(floor(bboxes[i][2]) - delta_y - ceil(bboxes[i][3]) +delta_y+20)
                assert frame.shape[1]> abs(floor(bboxes[i][0]) -delta_x - ceil(bboxes[i][1])+delta_x)
                
            except IndexError as e:
                print("Index error at idx {}, for total bboxes: {}".format(i , len(bboxes)))
                continue
            except StopIteration as e2:
                break
            except AssertionError as e3:
                # traceback.print_exc()
                print("Assertion error for image of size {}, got croppping dims h,w:{} {} ".format(frame.shape , abs(floor(bboxes[i][2]) - delta_y - ceil(bboxes[i][3]) +delta_y+20) ,abs(floor(bboxes[i][0]) - delta_x - ceil(bboxes[i][1])+delta_x)))
                frame = frame
                

            frame_cropped.append(frame)

        return frame_cropped

    def __getitem__(self, index) -> Any:
            
            segment_paths , label ,id_car= self.data[index]

            frames = [int(os.path.basename(x)[:-4]) for x in segment_paths]

            
            bboxes_frames=[]
            frames_annotated=[]


            #iter over frames in seg
            for i,frame_counter in (pbar2:=tqdm(enumerate(frames) , position=0,desc='Outter')):
                #iter over detections labels
                for detection_gt in (pbar:=tqdm(self.gt , position=1,desc='Inner')):
                    pbar2.set_description_str("now frames # {}".format(i))
                    
                    pbar.set_description_str("Labelled detections are {}".format(detection_gt))
                    if detection_gt[1]<0:continue
                    # print("Searching for car of id {}, now id in detectino is {}".format(id_car , detection_gt[1]))
                    ##find closest frame in detectios:
                    # if (frame_counter-detection_gt)<=5 or (detection_gt-frame_counter)<=5:

                    #     bboxes_frames.append((detection_gt[3],frame_counter - detection_gt)) ##append the bbox associated to that frame
                    #     bboxes_frames = sorted(bboxes_frames , key = bboxes_frames[1])
                    #     bboxes_frames = _remove_duplicate_bboxes(bboxes_frames)
                    #     bboxes_frames = [bboxes_frames[0]]
                    
                    if detection_gt[0] == frame_counter and id_car == detection_gt[1]: #check id car is the detected in labels.
                        
                        frames_annotated.append(detection_gt[0])
                        bboxes_frames.append([_x:=(detection_gt[2]),_x_w:=(detection_gt[2]+detection_gt[4]),
                                              _y:=(detection_gt[3]),_y_h:=(detection_gt[3]+detection_gt[5]) ])
                        
                        
                        break

                    else:
                        # raise Exception("unexpected consitenmcy mismath, got id car and id bbox: {} {}".format(id_car , j[1]))
                        continue
                        

            no_crop=False
            counter_missing=0
            for i,x in enumerate(bboxes_frames): 
                if x == [] :
                    counter_missing+=1
                    bboxes_frames[i] = [-1]*4

                else:
                    counter_missing=0
                
                
                    # raise Exception("Unexpected bbox dims")

            frame_stack = [cv2.imread(x) for x in segment_paths]

            if bboxes_frames==[]: 
                frames_cropped = frame_stack
                no_crop=True
                # raise Exception("No bboxes detected")

            if counter_missing>5:
                    raise Exception("Detected more 5 frames missing, Interpolating ...")
                    self._interpolate_frames(frame_stack , bboxes_frames)
                
            
            if not no_crop:
                frames_cropped = self.crop_frames(frame_stack , bboxes_frames)

            frame_stack = frames_cropped

            assert frame_stack!=[]
            print(len(frame_stack))
            for i,x in enumerate(frame_stack):
                try:

                    cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/debug/example_pic2_0{}.png".format(i) , x)
                except Exception as e:
                    continue

            frame_tensor = torch.stack([self.transform_roi(x) for x in frame_stack] , dim=1)
            frame_tensor=frame_tensor.type(torch.float)
            
            img=self.transform(cv2.imread(segment_paths[0])).cpu().permute((1,2,0)).numpy()
           
            # cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/example_pic2.png" , img)
            # input("waiting")

            # frame = cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame_tensor[:,:,:,].detach().cpu().numpy())
            # input("waiting")
            assert  frame_tensor.size(0) == 3
            assert frame_tensor.size(2) == self.roi_W
            assert frame_tensor.size(3) == self.roi_H

            
            label_tensor = torch.tensor(self.class_map[label] , dtype = torch.long)


            return frame_tensor , label_tensor
        
    def _interpolate_frames(frame_stack:List[torch.tensor],
                            bboxes_frames:List[torch.tensor],
                            h:int,
                            w:int,
                            delta_x:int , delta_y:int)->List[torch.tensor]:
        """
        Interpolate bbox for missing bbox annotaitons.
        For small intervals (5 frames) consider linear interpolation
        """
        frame_out = []
        for i, frame , bbox in enumerate(zip(frame_stack , bboxes_frames)):

            if bbox == [] and i>5:

                y_pred = 2*bbox[i-1][3] - bbox[i-2][1]  #linear extrapolant 
                x_pred = 2*bbox[i-1][1] - bbox[i-2][1]

                frame = frame[y_pred - delta_y , y_pred + h + delta_y +20 : x_pred - delta_x , x_pred + delta_x]
            
            frame_out.append(frame)

        return frame_out  

    def __len__(self):
        
        return len(self.data)
    

    def get_weights(self):
        return self.weights 
    
    def test_stuff(self,):
            """
            testing routing for debugging branch--not implemented
            """
            input("Total ds lenght is :{}".format(len(self.data)))
            for index in range(len(self.data)):
                segment_paths , label ,id_car= self.data[index]

                frames = [int(os.path.basename(x)[:-4]) for x in segment_paths]
                
                bboxes_frames=[]
                frames_annotated=[]


                #iter over frames in seg
                for frame_counter in frames:
                    #iter over detections labels
                    for detection_gt in self.gt:
                        if detection_gt[1]<0:continue
                        print("Searching for car of id {}, now id in detectino is {}".format(id_car , detection_gt[1]))
                        ##find closest frame in detectios:
                        # if (frame_counter-detection_gt)<=5 or (detection_gt-frame_counter)<=5:

                        #     bboxes_frames.append((detection_gt[3],frame_counter - detection_gt)) ##append the bbox associated to that frame
                        #     bboxes_frames = sorted(bboxes_frames , key = bboxes_frames[1])
                        #     bboxes_frames = _remove_duplicate_bboxes(bboxes_frames)
                        #     bboxes_frames = [bboxes_frames[0]]
                        if detection_gt[0] == frame_counter and id_car == detection_gt[1]:
                            
                            frames_annotated.append(detection_gt[0])
                            break

                        else:
                            # raise Exception("unexpected consitenmcy mismath, got id car and id bbox: {} {}".format(id_car , j[1]))
                            continue

                            
                
                for x in bboxes_frames: 
                    if x == []: break
                if bboxes_frames==[]: break

                # bboxes_frames = [list(filter(lambda x: x[0]==y, self.gt))[-4:] for y in frames_num]
                
                frame_stack = [cv2.imread(x) for x in segment_paths]

                frames_cropped = self.crop_frames(frame_stack , bboxes_frames)

                frame_stack = frames_cropped

                assert frame_stack!=[]
                
                for i,x in enumerate(frame_stack):
                    cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/debug/example_pic2_0{}.png".format(i) , x)
                    


                frame_tensor = torch.stack([self.transform_roi(x) for x in frame_stack] , dim=1)
                frame_tensor=frame_tensor.type(torch.float)
                
                img=self.transform(cv2.imread(segment_paths[0])).cpu().permute((1,2,0)).numpy()
            
                # cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/example_pic2.png" , img)
                # input("waiting")

                # frame = cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame_tensor[:,:,:,].detach().cpu().numpy())
                # input("waiting")
                assert  frame_tensor.size(0) == 3
                assert frame_tensor.size(2) == self.roi_W
                assert frame_tensor.size(3) == self.roi_H

                
                label_tensor = torch.tensor(self.class_map[label] , dtype = torch.long)


                return frame_tensor , label_tensor
        
class prevention_dataset_val(Dataset):
    """
    map style dataset:
    Every _PADDED_FRAMES frames -> one prediction (Lane Keep)
    Every delta frames (defined in lane_changes.txt) -> one pred (Lane change right/left)

    return 4d rgb tensor

    model always takes _PADDED frames and makes one prediction
    """

    def __init__(self,
                 root,
                 label_root,**kwargs
                 ) -> None:
        super().__init__()
        self.H = 600
        self.W = 600
        self.roi_H = 200
        self.roi_W = 200
        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame
        if "gt_root" in kwargs: self.gt_root = kwargs["gt_root"]

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)[0]
        self.gt = _read_labels_gt(labels_root = self.gt_root)

        self.class_map = { "LLC":0, "RLC":1}

        #assign lane change clases
        self.val_split = math.floor(len(self.labels)*0.8)-1

        self.labels = self.labels[self.val_split:]#train split 57 first maneuver-but need to assign last frame of previeous maneuver to set the index for lane keep in between data
        self.transform_roi = Compose([ ToImage() , CenterCrop((self.roi_H,self.roi_W)) ])


        for maneuver_info in self.labels:
            lane_start = maneuver_info[3]
            if lane_start<0:continue
            lane_end = maneuver_info[4]
            id_car = abs(maneuver_info[0])

            maneuver_event = maneuver_info[2]
            assert maneuver_event==3 or maneuver_event==4, "Expected maneuver label in 3,4, got {}".format(maneuver_event)
            # if lane_end-lane_start>72:continue  #skip maneuvers taking too long
            if maneuver_event==4: maneuver_event="RLC"
            if maneuver_event==3: maneuver_event="LLC"
            
            frames_paths = sorted([join(root , str(x) + ".png") for x in range(lane_start-5 , lane_end)])

            assert len(frames_paths)>0,"The expected frame range is {} {}".format(lane_start-5 , lane_end)
            
            self.data.append([frames_paths , maneuver_event , id_car])


        #assign lane keep clases
       
        i=self.labels[0][5]
        
          #assign lane keep clases
        if 0:
            lk_counter=0
            for maneuver in self.labels:
                
                frames_paths = sorted([join(root , str(x) + ".png") for x in range( i , maneuver[4])])
                
                for j in range(len(frames_paths)//_PADDED_FRAMES):
                    counter = 0
                    _temp_list = []
                    for frame in frames_paths:
                        _temp_list.append(frame)

                        if counter==_PADDED_FRAMES:
                            self.data.append([_temp_list , "LK"])
                            lk_counter+=1
                            break
                        else:
                            counter+=1
                    
                
                i=maneuver[5] #assingn next start to end of current manuver




        # for j in range(0,20,self._MAX_VIDEO_FRAMES):
        #     frames_temp = [] 
        #     for frame_path in glob(join( root, "*.png")):
        #         frames_temp.append(frame_path)
        #         if len(frames_temp)==20:break
                        
        #     for maneuver_info in self.labels:
        #         if frame_path in range(maneuver_info[5] - maneuver_info[4]):
            self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter=0,0,0
        # for item in self.data:
        #     if item[1]=="LK":self.lane_keep_counter+=1
        #     elif item[1]=="LLC":self.lane_change_right_counter+=1
        #     elif item[1]=="RLC":self.lane_change_left_counter+=1
        #     # print("Currently have {} lane keep, {} left lane change , {} right lane change".format(self.lane_keep_counter,self.lane_change_left_counter,self.lane_change_right_counter))
        
    def _print_stat(self):
        return "LK,LLC,RLC: {}/ {} / {}".format(self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter)

    def crop_frames(self,frames,bboxes):

        delta_y=20 #pixels left right tolerance
        delta_x = 30
        frame_cropped=[]
        
        for i,frame in enumerate(frames):
            try:
            
                if bboxes != [-1]*4:
                    frame = frame[floor(bboxes[i][2]) - delta_y :ceil(bboxes[i][3]) +delta_y+20 , floor(bboxes[i][0]) -delta_x:ceil(bboxes[i][1])+delta_x ] #crop
         
                elif bboxes == [-1]*4:
                    frame = frame

                assert frame.shape[0]> abs(floor(bboxes[i][2]) - delta_y - ceil(bboxes[i][3]) +delta_y+20)
                assert frame.shape[1]> abs(floor(bboxes[i][0]) -delta_x - ceil(bboxes[i][1])+delta_x)
                
            except IndexError as e:
                print("Index error at idx {}, for total bboxes: {}".format(i , len(bboxes)))
                continue
            except StopIteration as e2:
                break
            except AssertionError as e3:
                # traceback.print_exc()
                print("Assertion error for image of size {}, got croppping dims h,w:{} {} ".format(frame.shape , abs(floor(bboxes[i][2]) - delta_y - ceil(bboxes[i][3]) +delta_y+20) ,abs(floor(bboxes[i][0]) - delta_x - ceil(bboxes[i][1])+delta_x)))
                frame = frame
                

            frame_cropped.append(frame)

        return frame_cropped
    
    def _read_labels_gt(labels_root):
        """
        read car bbox detections
        """
        detections=[]
        df = pd.read_csv(labels_root, sep=" ")

        with open(labels_root , "r") as f:
            while True:
                line = f.readline()
                line = line[:-1]
                line_p=[]
                for x in line.rsplit(" "):
                    if x[1:-4]=="":continue

                    x= int(float(x[1:-4])*10**int(x[-1]))

                    line_p.append(x)
                line=line_p
                assert type(line)==list
                if not line:
                    break
                else:
                    detections.append(line)    #frame,id,x,y,width,height
        
        return detections

    def __getitem__(self, index) -> Any:
            
            segment_paths , label , id_car= self.data[index]

            frames = [int(os.path.basename(x)[:-4]) for x in segment_paths]

            bboxes_frames=[]
            frames_annotated=[]

            #iter over frames in seg-use detection from 1st frame to filter all next detections.
            #crop frames according to that detection-use in prediction and validation
            for i,frame_counter in (pbar2:=tqdm(enumerate(frames) , position=0,desc='Outter')):

                #iter over detections labels
                detection_id = 0

                for detection_gt in (pbar:=tqdm(self.gt , position=1,desc='Inner')):
                    
                    pbar2.set_description_str("now frames # {}".format(i))
                    
                    pbar.set_description_str("Labelled detections are {}".format(detection_gt))

                    if detection_gt[1]<0:continue

                    if (detection_gt[0] == frame_counter and id_car == detection_gt[1]): #check id car is the detected in labels.
                        
                        frames_annotated.append(detection_gt[0])  #append frame

                        bboxes_frames.append([_x:=(detection_gt[2]),_x_w:=(detection_gt[2]+detection_gt[4]),  ##bbox for that car
                                              _y:=(detection_gt[3]),_y_h:=(detection_gt[3]+detection_gt[5]) ])
                        

                        #break in 1st detection of same car for that frame
                        break

                    else:
                        #no bbox detection found for that frame and that car. Continue in next detection line
                        continue
                        
                


            no_crop=False
            counter_missing=0
            for i,x in enumerate(bboxes_frames): 
                if x == [] :
                    counter_missing+=1
                    bboxes_frames[i] = [-1]*4

                else:
                    counter_missing=0
                
                

            frame_stack = [cv2.imread(x) for x in segment_paths]

            if bboxes_frames==[]: 
                frames_cropped = frame_stack
                no_crop=True

            if counter_missing>5:
                    raise Exception("Detected more 5 frames missing, Interpolating ...")
                    self._interpolate_frames(frame_stack , bboxes_frames)
                
            if not no_crop:
                frames_cropped = self.crop_frames(frame_stack , bboxes_frames)

            frame_stack = frames_cropped

            assert frame_stack!=[]
        

            frame_tensor = torch.stack([self.transform_roi(x) for x in frame_stack] , dim=1) #preprocess cropped frames-resize to standard dims

            frame_tensor=frame_tensor.type(torch.float)
                    
            
            label_tensor = torch.tensor(self.class_map[label] , dtype = torch.long)

            #return croped frame tensor with one car detection

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)
    

class prevention_dataset_test(Dataset):
    """
    map style dataset:
    Every _PADDED_FRAMES frames -> one prediction (Lane Keep)
    Every delta frames (defined in lane_changes.txt) -> one pred (Lane change right/left)

    return 4d rgb tenso

    model always takes _PADDED frames and makes one prediction
    """

    def __init__(self,
                 root,
                 label_root,
                 ) -> None:
        super().__init__()
        self.H = 245
        self.W = 245
        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame


        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)[0]
        self.class_map = {"LK":2, "LLC":0, "RLC":1}

        #assign lane change clases

        self.labels = self.labels[:]#train split 57 first maneuver-but need to assign last frame of previeous maneuver to set the index for lane keep in between data


        for maneuver_info in self.labels:
            lane_start = maneuver_info[3]
            lane_end = maneuver_info[4]
            if lane_start<0:continue
            maneuver_event = maneuver_info[2]
            if maneuver_event!=3 and maneuver_event!=4:continue
            assert maneuver_event==3 or maneuver_event==4, "Expected maneuver label in 3,4, got {}".format(maneuver_event)
            # if lane_end-lane_start>72:continue  #skip maneuvers taking too long
            if maneuver_event==4: maneuver_event="RLC"
            if maneuver_event==3: maneuver_event="LLC"
            
            frames_paths = sorted([join(root , str(x) + ".png") for x in range(lane_start-5 , lane_end)])

            assert len(frames_paths)>0,"The expected frame range is {} {}".format(lane_start-5 , lane_end)
            
            self.data.append([frames_paths , maneuver_event])


        #assign lane keep clases
        i=self.labels[0][5]
        
          #assign lane keep clases
        if 0:
            lk_counter=0
            for maneuver in self.labels:
                
                frames_paths = sorted([join(root , str(x) + ".png") for x in range( i , maneuver[4])])
                
                for j in range(len(frames_paths)//_PADDED_FRAMES):
                    counter = 0
                    _temp_list = []
                    for frame in frames_paths:
                        _temp_list.append(frame)

                        if counter==_PADDED_FRAMES:
                            self.data.append([_temp_list , "LK"])
                            lk_counter+=1
                            break
                        else:
                            counter+=1
                    
                
                i=maneuver[5] #assingn next start to end of current manuver




        # for j in range(0,20,self._MAX_VIDEO_FRAMES):
        #     frames_temp = [] 
        #     for frame_path in glob(join( root, "*.png")):
        #         frames_temp.append(frame_path)
        #         if len(frames_temp)==20:break
                        
        #     for maneuver_info in self.labels:
        #         if frame_path in range(maneuver_info[5] - maneuver_info[4]):


    def __getitem__(self, index) -> Any:
            segment_paths , label = self.data[index]

            img = cv2.imread(segment_paths[0])
            
            frame_stack = [cv2.imread(x) for x in segment_paths]
            frame_tensor = torch.stack([self.transform(x) for x in frame_stack] , dim=1)

            frame = cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame_tensor[:,:,:,].detach().cpu().numpy())
            # cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame )

            # assert frame_tensor.size(1) == _PADDED_FRAMES and frame_tensor.size(0) == 3

            label_tensor = torch.tensor(self.class_map[label] , dtype = torch.float)

            print("val tensor dims are {}".format(frame_tensor))

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)

def construct_ds():
    """
    construct concat dataset from concatenation using torch.util.data.ConcatDataset
    
    """
    ds_out=[]
    for datasets in glob(join("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_0[2-5]/")):
        label=join(datasets , "processed_data" , "detection_camera1" , "lane_changes.txt")
        src = join(datasets , "segmented_frames/")
        ds = prevention_dataset_train(root= src , label_root= label)
        ds_out.append(ds)
    ds = ConcatDataset(ds_out)
    return ConcatDataset

class custom_concat_dataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, 
                 datasets: Iterable[Dataset],
                 ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._labels_dict={}

        for i,d in enumerate(self.datasets):
            
            self._labels_dict.update({"dataset_{}_{desc}".format(i , desc = d._get_desc() ):d._print_stat()})

    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def _get_ds_labels(self):
        return self._labels_dict

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



class union_prevention(prevention_dataset_train , Dataset):
    def __init__(self):
        # super(union_prevention , self).__init__()

        ds1=prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames",
                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")
        
        ds2=prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/segmented_frames",
                                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/processed_data/detection_camera1/lane_changes.txt")
        ds3=prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/segmented_frames",
                                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/processed_data/detection_camera1/lane_changes.txt")
        ds4=prevention_dataset_train(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_05/drive_03/segmented_frames"
                                                ,label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt") #r05_d03
        
        
        print("Train Dataset size is:\nDs1(rec01_01):{}\nDs2(rec02_01):{}\nDs3(rec03_01):{}\nDs4(rec05_03):{}\n".format(len(ds1),len(ds2),len(ds3),len(ds4)))
        print("Classes Ds1{}\nDs2{}\nDs3{}\nDs4{}\n".format(ds1._print_stat(),ds2._print_stat(),ds3._print_stat(),ds4._print_stat()))
        input("_____Printed stats_____________")

        ds5=prevention_dataset_val(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames",
                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")
        
        ds6=prevention_dataset_val(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/segmented_frames",
                                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/processed_data/detection_camera1/lane_changes.txt")
        ds7=prevention_dataset_val(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/segmented_frames",
                                                label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/processed_data/detection_camera1/lane_changes.txt")
        ds8=prevention_dataset_val(root= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_05/drive_03/segmented_frames"
                                                ,label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_05/drive_03/processed_data/detection_camera1/lane_changes.txt") #r05_d03
        
        
        
        print("val Dataset size is:\nDs1(rec01_01):{}\nDs2(rec02_01):{}\nDs3(rec03_01):{}\nDs4(rec05_03):{}\n".format(len(ds5),len(ds6),len(ds7),len(ds8)))
        print("Classes Ds1{}\nDs2{}\nDs3{}\nDs4{}\n".format(ds5._print_stat(),ds6._print_stat(),ds7._print_stat(),ds8._print_stat()))
        input("_____Printed stats_____________")

def _get_semented_data_paths()->List[Tuple]:
    _root_dir= "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/"
    paths=[]
    for recording in os.listdir(_root_dir):
        for drive in os.listdir(join(_root_dir,recording)):
            paths.append((join(_root_dir , recording, drive , "processed_data") , join(_root_dir , recording, drive , "video_camera1.mp4")))
    return paths

if __name__=="__main__":
    # data_labels_path = [
    #                     # ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/video_camera1.mp4") , 
    #                     ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/video_camera1.mp4"),
    #                     ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_04", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_04/video_camera1.mp4"),
    #                     # ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_05", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_05/'video_camera1(4).mp4'")
    #                     ]

    # dirs = 
    _r ,_d = 2 ,1 
    data_labels_path = [("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_03/drive_02/processed_data",
                        "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_03/drive_02/r3_d2_video.mp4")]
    _segment_video(data_path = data_labels_path)


