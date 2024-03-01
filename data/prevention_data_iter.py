from typing import Any, List, Tuple
from cv2 import VideoCapture
import torch
from torchvision.transforms import ToTensor
import  cv2
import queue
import traceback

class video_read_exception(Exception):

    def __init__(self, message, ) -> None:
        super().__init__(message, )


class read_frame_from_iter(torch.utils.data.IterableDataset):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection. 

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window


    """
    def __init__(self , 
                 path_to_video:str , 
                 path_to_label:str,
                 splits=[0.8,0.1,0.1]:List[float],
                 horizon=5 :int,
                 *args,
                 **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter).__init__()
        self.path_to_video = path_to_video
        self.path_to_lane_changes = path_to_label
          
        self.horizon = horizon
        self.transform = ToTensor()

        self._cap = VideoCapture(self.path_to_video)
        self._maneuver_type=[]

        self._length = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))   #used to assign timestamp iterato

        self._labels = queue.Queue()            #store the label data from prevention
        

        _lines=[]
        with open(self.path_to_lane_changes , "r") as f:   #read lane_change_file
                try:
                    _lines.append(f.readlines())
                except (StopIteration,EOFError) as e:
                    raise
        _lines=_lines[0]
        
        self._next_maneuver_begin=[]
        self._next_maneuver_end=[]
        #preprocess dataset. #rd item contains frame idx. second to last and first to last manever
        for line in _lines:

            _line = line[:-1]  #remove\n
            # self._labels.put((int(x) for x in _line.rsplit(" ")))  #make list of gt lines
            _line = [int(x) for x in _line.rsplit(" ")]
            
            self._next_maneuver_begin.append(_line[3])   #maneuver start times
            self._next_maneuver_end.append(_line[4])
            self._maneuver_type.append(_line[2])

            # self._next_step.put(labels[line+1][3])    #maneuver end times

        self._next_maneuver_begin = iter(self._next_maneuver_begin)
        self._next_maneuver_end = iter(self._next_maneuver_end)

        self._maneuver_type = iter(self._maneuver_type)
        self._current_timestep = iter(range(1, self._length))

            
    def __iter__(self):
        return self
    
    def _info_gen(self):
        """
        get current frame infor and update iterators
        """
        
        return next(self._current_timestep) , next(self._next_maneuver_begin) , next(self._next_maneuver_end) , next(self._maneuver_type)
    
    def __next__(self) -> Tuple[torch.FloatTensor , torch.FloatTensor]:
            """
            iterate over the video.
            Return video frame tensor (spatiotemporal) before maneuver (window frames)

            """

            # _maneuver = _current_step[5]
            # _end = _current_step[6]
            # _maneuver_type=_current_step[3]

            _current_timestep , _next_maneuver_begin , _next_maneuver_end ,  _manuever_type = self._info_gen()

            #iter over redundant video frames
            for _ in range(_current_timestep , _next_maneuver_begin - self.horizon ):
                _ = next(self._current_timestep) #update current time
                ret,_ = self._cap.read()  #move past redundant frames
                if not ret:
                    traceback.print_exc()
                    raise ValueError()

            try:
                frame_tensor = (self._get_video_tensor(delta := _next_maneuver_end - _next_maneuver_begin))
                assert len(frame_tensor) == 5+delta,"expected {} frames before prediction, got {}".format(5+delta,len(frame_tensor))
                frame_tensor = torch.stack([self.transform(x) for x in frame_tensor])  #apply to tensor
                label_tensor = torch.FloatTensor(_manuever_type)
            except StopIteration as e:
                traceback.print_exc()
                raise 
            
            #switch channel - time segment dimensions ( 1,2)
            frame_tensor = frame_tensor.permute((1,0,2,3))
            assert   frame_tensor.size(0)==3 and frame_tensor.size(1)>0 and frame_tensor.size(2) == 600 and frame_tensor.size(3)==1920,"got {} {} {} {}".format(frame_tensor.size(0),frame_tensor.size(1),frame_tensor.size(1),frame_tensor.size(1))

            return frame_tensor , label_tensor

    def _get_video_tensor(self , delta):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon + delta):
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