from cv2 import VideoWriter,imread,VideoCapture
import argparse
import cv2
from tqdm import tqdm
from os.path import join
def _preprocess_prevention(args):
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

    _preprocess_prevention(vars(parser.parse_args()))