
import cv2
from typing import Callable, List, Optional, Union
from matplotlib import pyplot as plt
import numpy as np
from torch import optim
import torch
from detectron2.structures import Boxes

def custom_err_handler(f):
    """
    custom err handling decorator
    """
    def wrapper(*args , **kwargs):
        try:
            f(*args , **kwargs)
        except ValueError as v:
            traceback.print_exc()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    return wrapper



def bisect_right(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo

class linear_warmup():
    """
    linear warmup
    """
    def __init__(self,opt , lr_init):
        # super(linear_warmup,self).__init__()
        self.opt=opt
        self.counter=0
        self.step()
        self.lr_init = lr_init
        self.last_lr=lr_init
        self.epoch=0
        self.Max_epochs = 10
    
    def get_last_lr(self):
        return self.last_lr

    def step(self):
        "Update parameters and rate"
        self._step += 1
        self.epoch+=1
        if self.epoch<self.Max_epochs:
            last_lr = self.get_last_lr()
        else:
            super(linear_warmup,self).step()
            pass
        
        lr = self.last_lr + self.rate()*self._step
   
        for p in self.optimizer.param_groups:
            p['lr'] = lr
       
        self.optimizer.step()

# def sort_torch_iterable_by_key(tensor_obj: torch.tensor, 
#                                key_obj: Callable)->Tuple[torch.tensor , Callable]:
    
    

#     tensor_key_tuple = list(zip(tensor , key))

def apply_bboxes_single(frames: List[np.array] , bboxes: List[Boxes],classes: List[str] , conf:List, delta_x = None , delta_y=None)->List[torch.tensor]:
        """
        apply one bbox for each frame-assumming 1st result of detectron2 will correspond to car making maneuver.
        """
        frame_seg=[]
        for frame ,bbox in zip(frames,bboxes):
            
            if bbox.tensor.numel()!=0:
                    bbox = list(map(lambda x: int(x), bbox.tensor[0]))
                    frame = frame[:,bbox[1] :bbox[3] , bbox[0]:bbox[2]]
            else:
                raise NotImplementedError
            
            frame_seg.append(frame)


        return frame_seg



        
def apply_bboxes(frames: List[torch.tensor], 
                 bbox: List[Boxes],
                 classes: List[str],
                 conf:List,
                 delta_x = None,delta_y=None)-> Union[ List[List[torch.tensor]] , List[torch.tensor]]:
    """
    apply bboxes estimated by detectron2 in validation loop 

    Return list of 1 (=segment length) of M tensors (m = detections) , only car detection considered
    """
    assert isinstance(bbox, list),"Excepcted bbox lsit per frame, got{}".format(type(bbox))
    len_bboxes = len(bbox)
    frame_out = []

    if all(list(map(lambda x : x == [] , bbox))):  #recheck
        raise MyCustomError("Found empty bboxes for frame stack ... terminating")
    else:

        
        for idx, (frame , bboxes ,  class_detect , conf) in enumerate(zip(frames , bbox, classes , conf)): #for all frames in current segment, and all bbox list on same segment
            
            bboxes:torch.tensor
            assert frame is not None and frame!=[]
            _nonempty_mask = bboxes.nonempty(threshold=0.0)

            if not any(_nonempty_mask):  #if no boxes found in frame - caught by wrrapper func
                frame_out.append(cv2.resize(frame, dsize=(200,200) , interpolation=cv2.INTER_AREA))
                # raise MyCustomError("Found empty bboxes for singular frme of video segment stack ... terminating/inteprolate?") 
                
            elif any(_nonempty_mask):
                detections = 0
                max_detections=4
                detections_vect = zip( bboxes , conf)

                detections = (sorted(detections_vect , key = lambda x : x[1]))[:max_detections]

                _nonempty_mask_sorted = detections[0].nonempty()


                for i ,(bbox, confidences) in enumerate(detections_vect):   #tuple bbox and conf in single frame-can be many
                    frame_temp=[]
                    frame_temp_copy= []
                    if _nonempty_mask_sorted[i]==1:
                        
                        #first value->highest score corresponding to detected class
                        print("Detected class is {} with confidence {}".format(clas,conf))

                        #check if detection is member of class (car,bus,truck) and prediction conf is high
                        if  check_detections_category(clas)!=None and delta_x == None and delta_y==None and confidences>0.8: #prediction conf of single detection at image == frame
                            
                            frame_temp_copy = frame.copy()
                            input(bbox)
                            frame_temp_copy = frame_temp_copy[bbox[2] :bbox[3] , bbox[0]:bbox[1]]

                        else:
                            frame_temp_copy = frame.copy()
                            frame_temp_copy = cv2.resize(frame_temp_copy,dsize=(200,200),interpolation=cv2.INTER_AREA)

                        frame_temp.append(torch.tensor(frame_temp_copy))
                    else:
                        # raise NotImplementedError
                        frame_temp.append(torch.tensor(frame))


                    frame_temp = torch.stack([x for x in frame_temp])
                    frame_out.append(frame_temp)

        try:
            frame_out = torch.tensor(frame_out)
        except ValueError as e:
            frame_out=torch.stack([x for x in frame_out])
        except Exception as e:
            raise MyCustomError("detected objects dont pass threshold")
        return frame_out.unsqueeze(0)

class MyCustomError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

class NoBBOXFrame(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


def check_detections_category(idx:int)->Optional[str]:
    for k,v  in {"car":3 , 'motorcycle':4, 'truck': 8 ,'bus':6}.items():
        if idx==v:return k

    return None