import cv2
import argparse
def _read_lane_change_labels(label_root):
    """
    read lanechanges.txt,
    return maneuver_info: List[List[int]]
    """
    with open(label_root , "r") as labels:
        annotations = labels.readlines()
    for i,maneuver_info in enumerate(annotations):
        annotations[i] = maneuver_info[:-1] #rmv newline char
        annotations[i] = [int(x) for x in maneuver_info.rsplit(" ")] #extract info as list of list of int
    
    return maneuver_info

def main(**kwargs):
    labels = _read_lane_change_labels(kwargs["label_root"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap_in = cv2.VideoCapture(kwargs["video_root"] )
    fps = cap_in.get(cv2.CAP_PROP_POS_FRAMES)
    H,W = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT ),cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)

    cap_out = cv2.VideoWriter(kwargs["video_root"] , fourcc , fps , (W,H))

    _segment_interval_min=5
    
    

if __name__== "__main__":
    var = argparse.ArgumentParser()

    var.add_argument("--video_root",default="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4")
    main(vars(var.parse_args()))