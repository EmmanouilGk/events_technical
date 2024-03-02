import os
import sys
from cv2 import VideoWriter,imread,VideoCapture
import argparse
import cv2
from tqdm import tqdm
from os.path import join
import traceback

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

def _remove_missing_frames_(video_root = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                            label_root = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt", 
                            video_dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera_preprocessed.mp4",
                            label_dspt = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt"):
    """
    Remove corrupt frames from video, and decrement all frame idx in label_root (lane__changes.txt) (since total_frames(new) = total_frames(old) - corrupt_frames)
    and store new clean video
    """
    if os.path.isfile(video_dstp): os.remove(video_dstp) #remv existing preprocessed file
    cap = cv2.VideoCapture(video_root)   
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _total_corrupt_frames= 0 
    _sec_total = frames_total/fps
    H,W=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Frame size {}x{}".format(H,W))

    labels = _read_lane_change_labels(label_root)
    
    cap_out = cv2.VideoWriter(video_dstp , cv2.VideoWriter_fourcc(*"mp4v") , fps , (W,H)) #output for processed labels

    while True:
        try:
            ret,frame = cap.read()
            print("Now processing frame #{:>40f}/{:f}, second: {:>15f}/{:f}".format(_current_frame:=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) , frames_total,
                                                                                    _current_frame/fps , _sec_total ))

            if _current_frame <= frames_total:
                if not ret:
                    ###decrement labels
                    _total_corrupt_frames+=1
                    for annotation in labels:
                        input(annotation)
                        annotation[3]-=1 #manuevr start
                        annotation[4]-=1 #manuevr time
                        annotation[5]-=1 #maneuvr end
                else:
                    cap_out.write(frame)
            else:
                break

        except Exception as e:
            traceback.print_exc()

    #write new labels
    with open(label_dspt , "w") as label_out:

        _annotations_out=[]
        for annotation in labels:
            _temp = [str(x).zfill(6)+" " for x in annotation]
            _temp_2 = [[].join(x) for x in _temp]
            _temp_2 = _temp_2.join("\n")
            _annotations_out.append(annotation)
            
        label_out.writelines(_annotations_out)

    #validate new labels
    print("Found {} corrupt frames".format(_total_corrupt_frames))


def _preprocess_prevention(args: argparse.ArgumentParser):
    root = "/home/iccs/Desktop/isense/events/intention_prediction"
    video_src = join(root,"processed_data","video_camera1.mp4")

    video_train_dstp = join(root, "processed_data" , "video_train.avi" )
    video_val_dstp = join(root, "processed_data" , "video_val.avi" )
    video_test_dstp = join(root, "processed_data" , "video_test.avi" )

    cap = VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret,frame = cap.read() #get one forame for setting h,w in writer
    H, W = frame.shape[0] , frame.shape[1]

    input(H)
    input(W)

    cap_train = VideoWriter(video_train_dstp , cv2.VideoWriter_fourcc(*"MPEG") , fps , (W , H ))
    
    for _ in tqdm(range(1 , int(length*args["train"]))):
        ret, frame = cap.read()
        if not ret:
            raise Exception
        """
        ADD ANY PREPROCESSING HERE FOR VIDEO-LEVEL
        """
        frame = cv2.resize(frame , (W , H))
        cap_train.write(frame)

    cap_val = VideoWriter(video_val_dstp , cv2.VideoWriter_fourcc(*"MPEG") , fps , (W , H ))

    for _ in tqdm(range(1 , int(length*args["val"]))):
        ret, frame = cap.read()
        if not ret:
            raise Exception
        """
        ADD ANY PREPROCESSING HERE FOR VIDEO-LEVEL
        """
        frame = cv2.resize(frame , (W , H))
        cap_val.write(frame)

    cap_test = VideoWriter(video_test_dstp , cv2.VideoWriter_fourcc(*"MPEG") , fps , (W , H ))

    for _ in tqdm(range(1 , int(length*args["test"]))):
        ret, frame = cap.read()
        if not ret:
            raise Exception
        """
        ADD ANY PREPROCESSING HERE FOR VIDEO-LEVEL
        """
        frame = cv2.resize(frame , (W , H))
        cap_test.write(frame)

if __name__=="__main__":
    parser=  argparse.ArgumentParser()
    parser.add_argument("--train" , default=0.8)
    parser.add_argument("--val" , default=0.1)
    parser.add_argument("--test" , default=0.1)


    _var = vars(parser.parse_args())

    assert _var["train"]+_var["val"]+_var["test"]==1,"Expected sum of splits 1 got smth else"

    args = vars(parser.parse_args())

    if 1:
        _remove_missing_frames_()
        sys.exit(0)

    _preprocess_prevention(args)
