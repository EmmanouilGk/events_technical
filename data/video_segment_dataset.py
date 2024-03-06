import os
from typing import Any
import cv2
import sys
import torch
from torch.utils.data import Dataset, ConcatDataset
from os.path import join
from glob import glob
from torchvision.transforms import Compose , Resize,  ToTensor
import fnmatch
from ..conf.conf_py import _PADDED_FRAMES

from tqdm import tqdm

def compute_weights(ds, 
                    save = True):
           """
           Compute balancing wegiths for torch.utils.BatchSampler 
           
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
           class_counts = list((count_lk , count_llc , count_rlc))
           num_samples = sum(class_counts)
           class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
           weights = [class_weights[labels[i]] for i in range(int(num_samples))]
           weights= torch.DoubleTensor(weights)

           if save:
               torch.save(weights, "/home/iccs/Desktop/isense/events/intention_prediction/data/weights_torch/weights_union_prevention2.pt")


           return weights , class_weights


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

def _segment_video(src="/home/iccs/Desktop/isense/events/intention_prediction/processed_data/video_camera1.mp4",
                   dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/segmented_frames/",
                   **kwargs):
    
    if kwargs["data_path"]:
        for i, src in enumerate(kwargs["data_path"]):
            src = src[1]
            i=i+2
            dstp =  "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/"
            if not os.path.isdir(join(dstp,"segmented_test_frames")):os.mkdir(join(dstp,"segmented_test_frames"))
            dstp = join(dstp,"segmented_test_frames")
            print(src)
            cap = cv2.VideoCapture(src)
            frame_idx=0
            with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
                pbar.set_description_str("processing video-labels {}".format(os.path.basename(src)))
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
                 root:str,
                 label_root:str) -> None:
        super().__init__()
        self.H = 400
        self.W = 400
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

            for x in frames_paths: 
                assert os.path.isfile(x),"No file/bad file found at path {}".format(x)
            
            self.data.append([frames_paths , maneuver_event])
            

        if 1:
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
        self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter=0,0,0
        for item in self.data:
            if item[1]=="LK":self.lane_keep_counter+=1
            elif item[1]=="LLC":self.lane_change_right_counter+=1
            elif item[1]=="RLC":self.lane_change_left_counter+=1
            print("Currently have {} lane keep, {} left lane change , {} right lane change".format(self.lane_keep_counter,self.lane_change_left_counter,self.lane_change_right_counter))
        


    def __getitem__(self, index) -> Any:
            segment_paths , label = self.data[index]
            
            frame_stack = [cv2.imread(x) for x in segment_paths]

            frame_tensor = torch.stack([self.transform(x) for x in frame_stack] , dim=1)
            

            # img=self.transform(cv2.imread(segment_paths[0])).cpu().permute((1,2,0)).numpy()
           
            # cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/example_pic2.png" , img)
            # input("waiting")

            # frame = cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame_tensor[:,:,:,].detach().cpu().numpy())
            # input("waiting")
            assert  frame_tensor.size(0) == 3
            assert frame_tensor.size(2) == self.H
            assert frame_tensor.size(3) == self.W

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
                 label_root,
                 ) -> None:
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
            lane_start = maneuver_info[3]
            lane_end = maneuver_info[4]
            
            maneuver_event = maneuver_info[2]
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

            # assert frame_tensor.size(1) == _PADDED_FRAMES and frame_tensor.size(0) == 3

            label_tensor = torch.tensor(self.class_map[label] , dtype = torch.float)

            return frame_tensor , label_tensor
        
        
    def __len__(self):
        return len(self.data)
    

class prevention_dataset_test(Dataset):
    """
    map style dataset:
    Every _PADDED_FRAMES frames -> one prediction (Lane Keep)
    Every delta frames (defined in lane_changes.txt) -> one pred (Lane change right/left)

    return 4d rgb tensor

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
        
        self.labels = _read_lane_change_labels(label_root)
        self.class_map = {"LK":0, "LLC":1, "RLC":2}

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


class union_prevention(prevention_dataset_train , Dataset):
    def __init__(self):
        super(union_prevention , self).__init__()



if __name__=="__main__":
    # data_labels_path = [
    #                     # ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_02/video_camera1.mp4") , 
    #                     ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_03/video_camera1.mp4"),
    #                     ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_04", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_04/video_camera1.mp4"),
    #                     # ("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_05", "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/processed_data_05/'video_camera1(4).mp4'")
    #                     ]
    data_labels_path = [("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_04/drive_03/processed_data/",
                        "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_04/drive_03/video_camera1.mp4")]
    _segment_video(data_path = data_labels_path)