
from typing import List
from matplotlib import pyplot as plt
from torch import optim
import torch

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
        
def apply_bboxes(frames: List[torch.tensor], 
                 bbox: List[torch.tensor],
                 classes: List[str],
                 delta_x = None,delta_y=None)->List[List[torch.tensor]]:
    """
    apply bboxes estimated by detectron2 in validation loop 

    Return list of N lsits (=segment length) of M tensors (m = detections) , only car detection considered
    """

    frame_out = []
    for frame , bboxes ,  class_detect in zip(frames , bbox, classes): #for all frames in current segment, and all bbox list on same segment
        if bboxes ==[] :raise MyCustomError("Found empty bboxes for class car object ... terminating")
        if bboxes!=[]:
            frame_temp=[]
            for bbox in bboxes:
                frame_temp= []
                print("Detected class is {}".format(class_detect))
                if class_detect =="car" and delta_x == None and delta_y==None:
            
                    frame = frame[bboxes[1] :bboxes[1]+bboxes[3] , bboxes[2]:bboxes[2]+bboxes[4]]

                frame_temp.append(frame)

            frame_temp =torch.tensor(frame_temp)
            frame_out.append(frame_temp)


    frame_out = torch.tensor(frame_out)
    return torch.tensor(frame_out).unsqueeze(0)

class MyCustomError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)