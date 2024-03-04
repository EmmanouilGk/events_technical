from typing import Any
import cv2
from torch.utils.data import Dataset
from os.path import join
from glob import glob

def _segment_video(src="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames/"):
    cap = cv2.VideoCapture(src)
    frame_idx=0
    while True:
        if cap.isOpened():
            ret,frame = cap.read()
            dstp_frame = join(dstp , frame_idx + ".png")
            cv2.imwrite(dstp_frame , frame)
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

class prevention_dataset(Dataset):
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

    def __getitem__(self, index) -> Any:
        
    def __len__

