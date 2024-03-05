import os
from typing import Any
import cv2
import torch
from torch.utils.data import Dataset
from os.path import join
from glob import glob
from torchvision.transforms import Compose , Resize,  ToTensor
from ..conf.conf_py import _PADDED_FRAMES

from tqdm import tqdm

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


    return {"LLC" : w_llc,
            "RLC": w_rlc,
            "LK": w_lk}

def _segment_video(src="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames/"):
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

    with open(label_root , "r") as labels:
        annotations = labels.readlines()
    for i,maneuver_info in enumerate(annotations):
        annotations[i] = maneuver_info[:-1] #rmv newline char
        annotations[i] = [int(x) for x in maneuver_info.rsplit(" ")] #extract info as list of list of int
        maneuver_sequences.append(annotations[i][:])  #append maneuver info
        
        
    return maneuver_sequences

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
                 root,
                 label_root) -> None:
        super().__init__()
        self.H = 256
        self.W = 256
        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)
        self.class_map = {"LK":0, "LLC":1, "RLC":2}

        #assign lane change clases
        
        self.labels = self.labels[:57]#train split
   
        for maneuver_info in self.labels:
            lane_start = maneuver_info[3]
            if lane_start<0: continue
            lane_end = maneuver_info[4]
            maneuver_event = maneuver_info[2] #maneuver type
            # if lane_end-lane_start>72:continue  #skip maneuvers taking too long
            if maneuver_event==4: maneuver_event="RLC"
            if maneuver_event==3: maneuver_event="LLC"
            

            assert lane_start>0 and lane_end>0, "Expected positive maneuver labels,got {} {}".format(lane_start , lane_end)

            frames_paths = sorted([join(root , str(x) + ".png") for x in range(lane_start-5 , lane_end)])

            for x in frames_paths: assert os.path.isfile(x),"No file/bad file found at path {}".format(x)
            
            self.data.append([frames_paths , maneuver_event])
            


        #assign lane keep clases
        i=0
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
            
        self.weights=_calculate_weigths_classes(maneuvers=self.labels, lk=lk_counter)




    def __getitem__(self, index) -> Any:
            segment_paths , label = self.data[index]
            
            frame_stack = [cv2.imread(x) for x in segment_paths]
            frame_tensor = torch.stack([self.transform(x) for x in frame_stack] , dim=1)

            assert  frame_tensor.size(0) == 3

            label_tensor = torch.tensor(self.class_map[label] , dtype = torch.long)

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)
    

    def get_weights(self):
        return self.weights 
    
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
                 label_root) -> None:
        super().__init__()
        self.H = 256
        self.W = 256
        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)
        self.class_map = {"LK":0, "LLC":1, "RLC":2}

        #assign lane change clases

        self.labels = self.labels[57 - 1:65]#train split 57 first maneuver-but need to assign last frame of previeous maneuver to set the index for lane keep in between data
   
        for maneuver_info in self.labels:
            lane_start = maneuver_info[4]
            lane_end = maneuver_info[5]
            maneuver_event = maneuver_info[3]
            # if lane_end-lane_start>72:continue  #skip maneuvers taking too long
            if maneuver_event==3: maneuver_event="RLC"
            if maneuver_event==2: maneuver_event="LLC"
            
            frames_paths = sorted([join(label_root , str(x) + ".png") for x in range(lane_start-5 , maneuver_event)])
            
            self.data.append([frames_paths , maneuver_event])


        #assign lane keep clases
        i=self.labels[0][5]
          #assign lane keep clases
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
            
            frame_stack = [cv2.imread(x) for x in segment_paths]
            frame_tensor = torch.stack([self.transform(x) for x in frame_stack] , dim=1)

            assert frame_tensor.size(1) == _PADDED_FRAMES and frame_tensor.size(0) == 3

            label_tensor = torch.tensor(self.class_map(label) , dtype = torch.float)

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)
    
