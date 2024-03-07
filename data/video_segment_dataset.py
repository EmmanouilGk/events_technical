import os
from typing import Any, List, Tuple
import cv2
import math
import sys
import torch
from torch.utils.data import Dataset, ConcatDataset
from os.path import join
from glob import glob
from torchvision.transforms import Compose , Resize,  ToTensor , CenterCrop
import fnmatch
from ..conf.conf_py import _PADDED_FRAMES

from torchvision.transforms.v2 import ToImage

from tqdm import tqdm

def compute_weights(ds, 
                    save = True,
                    custom_scaling:int =1):
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
           class_counts = list((count_lk , custom_scaling*count_llc , custom_scaling*count_rlc)) #extra scaling for minority class
           num_samples = sum(list((count_lk , custom_scaling , count_rlc)))
           class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
           weights = [class_weights[labels[i]] for i in range(int(num_samples))]
           weights= torch.DoubleTensor(weights)

           if save:
               torch.save(weights, "/home/iccs/Desktop/isense/events/intention_prediction/data/weights_torch/weights_union_prevention3.pt")


           weight_dict = {"LK" : weights[0] , "LLC":weights[1], "RLC":weights[2]}
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
        self.H = 650
        self.W = 650

        self.transform = Compose([ ToImage() , CenterCrop((self.H,self.W)) ]) #transfor for each read frame

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)
        self.class_map = {"LK":0, "LLC":1, "RLC":2}

        #assign lane change clases

        self.train_split = math.floor(len(self.labels)*0.8)

        
        self.labels = self.labels[:self.train_split]#train split
   
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
            # print("Currently have {} lane keep, {} left lane change , {} right lane change".format(self.lane_keep_counter,self.lane_change_left_counter,self.lane_change_right_counter))


        
    def _print_stat(self):
        return "LK,LLC,RLC: {}/ {} / {}".format(self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter)

        

    def __getitem__(self, index) -> Any:
            segment_paths , label = self.data[index]
            
            frame_stack = [cv2.imread(x) for x in segment_paths]

            # cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/example_pic2_01.png" , frame_stack[0])
            # input("waiting")

            frame_tensor = torch.stack([self.transform(x) for x in frame_stack] , dim=1)
            

            img=self.transform(cv2.imread(segment_paths[0])).cpu().permute((1,2,0)).numpy()
           
            # cv2.imwrite("/home/iccs/Desktop/isense/events/intention_prediction/example_pic2.png" , img)
            # input("waiting")

            # frame = cv2.imwrite(" /home/iccs/Desktop/isense/events/intention_prediction/example_pic.png",frame_tensor[:,:,:,].detach().cpu().numpy())
            # input("waiting")
            assert  frame_tensor.size(0) == 3
            assert frame_tensor.size(2) == self.H
            assert frame_tensor.size(3) == self.W

            frame_tensor = torch.tensor(frame_tensor, dtype = torch.float)
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
        self.H = 600
        self.W = 600
        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame

        self.data=[]
        for video_frame_srcp in sorted(glob(join(root,".png"))):
            self.data.append(video_frame_srcp)
        
        self.labels = _read_lane_change_labels(label_root)
        self.class_map = {"LK":0, "LLC":1, "RLC":2}

        #assign lane change clases
        self.val_split = math.floor(len(self.labels)*0.8)-1

        self.labels = self.labels[self.val_split:]#train split 57 first maneuver-but need to assign last frame of previeous maneuver to set the index for lane keep in between data
   

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
            self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter=0,0,0
        for item in self.data:
            if item[1]=="LK":self.lane_keep_counter+=1
            elif item[1]=="LLC":self.lane_change_right_counter+=1
            elif item[1]=="RLC":self.lane_change_left_counter+=1
            # print("Currently have {} lane keep, {} left lane change , {} right lane change".format(self.lane_keep_counter,self.lane_change_left_counter,self.lane_change_right_counter))
        
    def _print_stat(self):
        return "LK,LLC,RLC: {}/ {} / {}".format(self.lane_keep_counter,self.lane_change_right_counter,self.lane_change_left_counter)


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

    data_labels_path = [("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_04/drive_01/processed_data/",
                        "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/new_data/recording_04/drive_01/r4_d1_video.mp4")]
    _segment_video(data_path = data_labels_path)