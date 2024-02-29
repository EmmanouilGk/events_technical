from typing import Any
from cv2 import VideoCapture
import torch
from torchvision.transforms import ToTensor
import  cv2
import queue

class video_read_exception(Exception):

    def __init__(self, message, ) -> None:
        super().__init__(message, )


class read_frame_from_iter():
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection. 

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window


    """
    def __init__(self , 
                 path_to_video , 
                 path_to_label,
                 
                 horizon=5,*args) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        self.path_to_video = path_to_video
        self.path_to_lane_changes = path_to_label
          
        self.horizon = horizon
        self.transform = ToTensor()

        self._cap = VideoCapture(self.path_to_video)
        self._current_step=0

        self._length = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))   #used to assign timestamp iterato
        self_timestamp_iter = iter(range(self._length))  #used to iterate over video and skip redundant frames
        self._labels = queue.Queue()            #store the label data from prevention

        _labels=[]
        with open(self.path_to_lane_changes , "r") as f:   #read lane_change_file
            while True:
                try:
                    labels = labels.append(f.readline())
                except StopIteration as e:
                    break
        
        #preprocess dataset. #rd item contains frame idx. second to last and first to last manever
        for line in range(len(labels)-1):
            _line = line[:-1]  #remove\n
            _labels.put((int(x) for x in line.rsplit(" ")))  #make list of gt lines
            
            # self._current_step.put(labels[line][3])   #maneuver start times
            # self._next_step.put(labels[line+1][3])    #maneuver end times
            
    def __iter__(self):
        return self
    
    def __next__(self):
            """
            iterate over the video.
            Return video frame tensor (spatiotemporal) before maneuver (window frames)

            """
            _current_entry = self._labels.get()
            _current_step = next(self._timestamp_iter)
           
            _begin = _current_entry[4]
            _maneuver = _current_entry[5]
            _end = _current_entry[6]
            _maneuver_type=_current_entry[3]

            #iter over redundant video frames
            for _ in range(_current_step , _begin - self.horizon ):
                _current_step = next(self._timestamp_iter)
                _,_ = self._cap.read()

            try:
                frame_tensor = (self._get_video_tensor_gen(self,maneuver_step = _maneuver,
                                                                end_step = _end))
                                    
                frame_tensor = self.transform(frame_tensor)
                label_tensor = torch.FloatTensor(_maneuver_type)
            
                
            except StopIteration as e:
                raise 

            yield frame_tensor , label_tensor

    def _get_video_tensor(self , maneuver_step , end_step):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon):
            #read 5 frames
            ret,val = self._cap.read()
            if ret:
                _frames.append(val)
            else:
                raise video_read_exception("Problem reading frame {}".format(j))
        
        return _frames

def main():
    pass

if __name__=="__main__":
    main()