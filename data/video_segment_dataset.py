import os
from typing import Any
import cv2
import torch
from torch.utils.data import Dataset
from os.path import join
from glob import glob

from tqdm import tqdm

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
        maneuver_sequences.append([annotations[i][3:6]])  #append maneuver info
        
    return maneuver_sequences

class base_dataset():

    def __init__(self,
                 root,
                 label_root) -> None:

            self.labels=_read_lane_change_labels(label_root)
            
            

class prevention_dataset_train(Dataset):
    """
    map style dataset:
    Every 20 frames -> one prediction (Lane Keep)
    Every delta frames (defined in lane_changes.txt) -> one pred (Lane change right/left)

    return 4d rgb tensor
    """

    def __init__(self,
                 root,
                 label_root) -> None:
        super().__init__()

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        

        self.labels = _read_lane_change_labels(label_root)


        #assign lane change clases
        maneuver_frames = []
        for maneuver_info in self.labels:
            lane_start = maneuver_info[4]
            lane_end = maneuver_info[5]
            maneuver_event = maneuver_info[3]
            if maneuver_event==3: maneuver_event="RLC"
            if maneuver_event==2: maneuver_event="LLC"
            
            frames_paths = sorted([join(label_root , x + ".png") for x in range(lane_start-5 , maneuver_event)])
            
            for maneuver_frame_path in frames_paths: 
                self.data.append([maneuver_frame_path , maneuver_event])


        #assign lane keep clases
        i=0
        for maneuver in self.labels:
            
            frames_paths = sorted([join(label_root , x + ".png") for x in range( i , maneuver[4])])
            for j in range(len(frames_paths)//20):
                counter = 0
                for frame in frames_paths:
                    
                    self.data.append([frame , "LK"])
                    if counter==20:
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
            frame_tensor = torch.stack([self.transforms(x) for x in frame_stack])

            label_tensor = torch.tensor(label , dtype = torch.float)

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)

def main():
    _segment_video()

if __name__=="__main__":
    main()