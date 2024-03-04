import cv2
import argparse

def _vid_gen(cap):
    while True:
        ret,val=cap.read()
        curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret:
            raise Exception
        yield val,curr_frame,curr_frame-5

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


def _write_lane_change( maneuver_sequences  ,
                       dstp , 
                       cap_in,
                       **kwargs):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap_out = cv2.VideoWriter(dstp , fourcc , fps=kwargs["fps"] , frameSize= (kwargs["W"] , kwargs["H"]))
    
    for maneuver in maneuver_sequences:
        lane_change_start = maneuver[3]
        lane_change_event = maneuver[4]
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, lane_change_start-5)
        for _ in range(lane_change_event - lane_change_start + 5 ):
            frame,_ = cap_in.read()
            cap_out.write(frame)

def _write_lane_keep(maneuver , dstp_root,cap_in):
    frame_list=[]
    for frame,idx in _vid_gen(cap_in):
        frame_list.write(frame)
        if maneuver[4]==idx:break

def _gen_pairs_man(man_src , ):
    """
    generate pairs of annotattions (maneuvers) (to iterate over)
    """
    man_processed = _read_lane_change_labels(man_src)
    man_total = len(man_processed)
    for idx_1,idx_2 in zip(range(0,man_total-1)  , range(1,man_total)):
        yield man_processed[idx_1 , :] , man_processed[idx_2,:]

def _get_seg_man(cap):
    """
    get sequences betwween maneuvers
    """
    for i, (frame_start , frame_end) in enumerate(_gen_pairs_man()):
        cap.set(cv2.CAP_PROP_POS_FRAMES , frame_start[5])
        while True:
            frame,idx=_vid_gen(cap)
            if idx == frame_end[4] - 5:
                break
            else:
                frame_list.append(frame)
                if len(frame_list)==20:
                    yield frame_list
                    frame_list=  []

def _save_seg(cap , dstp , **kwargs):
    """
    save seg
    """
    for frame_seg in _get_seg_man(cap):
        cap_out = cv2.VideoWriter(dstp , fourcc= cv2.VideoWriter_fourcc(*"mp4v"), fps = kwargs["fps"], frameSize=(kwargs["W"],kwargs["H"]))
        for frame in frame_seg:
            cap_out.write(frame)

def main():
    cap = cv2.VideoCapture(root = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                           )
    fps=  cap.get(cv2.CAP_PROP_FPS)
    H,W = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) , cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    _save_seg(cap , 
              dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1_seg",
              fps = fps, H= H , W = W)




def main(**kwargs):
    labels = _read_lane_change_labels(kwargs["label_root"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap_in = cv2.VideoCapture(kwargs["video_root"] )
    fps = cap_in.get(cv2.CAP_PROP_POS_FRAMES)
    H,W = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT ),cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)


    
    maneuver_sequences = _read_lane_change_labels(label_root="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt")

    #handle maneuver (label = 0/1) events
    for maneuver in maneuver_sequences:
        _write_lane_keep(maneuver , dstp_root = "")  #write lane change events until the maneuver in loop
        _write_lane_change(maneuver , dstp = "")  #write one lane change from one labels.txt entry

    #handle non-maneuver (label=2) events
    
    
    _segment_interval_len=5

    _frame_list = []

    _next_maneuver=next(maneuver_sequences_iter)
    for frame,idx,horizon_index in _vid_gen(cap_in):
        if idx != _next_maneuver:
            _frame_list.append(frame)
        if idx==_next_maneuver:


        if len(_frame_list)==_segment_interval_len:
            ##save frames and label
        
        


if __name__== "__main__":
    var = argparse.ArgumentParser()

    var.add_argument("--video_root",default="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4")
    main(vars(var.parse_args()))