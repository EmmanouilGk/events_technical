import logging
from typing import Any, List, Tuple
from cv2 import VideoCapture
import torch
from torchvision.transforms import ToTensor , Resize , Compose
import  cv2
import queue
import traceback

class video_read_exception(Exception):

    def __init__(self, message, ) -> None:
        super().__init__(message, )

class base_class_prevention(torch.utils.data.IterableDataset):
    """
    base class for prevention dataset in torch 
    parent of train,val,test
    """
    def __init__(self,path_to_video:str , 
                    path_to_label:str, 
                    prediction_horizon:int ,
                    splits: Tuple[float]):
        """
        init iterators to original videos and pointers to starting frames of val,test sets - used by inherited
        class members
        """
        super(base_class_prevention).__init__()
        train , val , test = splits
        self._train , self._val , self._test = 1 , 1-val , 1-test
        self.path_to_video = path_to_video
        self.path_to_lane_changes = path_to_label
        self.H = 400
        self.W = 800
        self.horizon = prediction_horizon


        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame


        self._cap_train = VideoCapture(self.path_to_video)
        self._max_train_frames = int(self._cap_train.get(cv2.CAP_PROP_FRAME_COUNT))   #MAX_FRAMES - used for timestamp iterator

        self._cap_val = set(cv2.CV_CAP_PROP_POS_FRAMES, (1-val)*self._max_train_frames)
        self._cap_test = set(cv2.CV_CAP_PROP_POS_FRAMES, (1-test)*self._max_train_frames)

    
        self._maneuver_type=[]

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
            _line = [int(x) for x in _line.rsplit(" ")]
            self._next_maneuver_begin.append(_line[3])   #maneuver start times-check and is correct 
            self._next_maneuver_end.append(_line[4])
            self._maneuver_type.append(_line[2])

            # self._next_step.put(labels[line+1][3])    #maneuver end times
        
class read_frame_from_iter_train(base_class_prevention):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , 
                 
                 splits: List[float] =[0.8,0.1,0.1] ,
                 horizon: int = 5 ,
                 *args,
                 **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter_train).__init__()
        
        self.video_iter = self._cap_train

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
                    # raise ValueError()
            try:
                frame_tensor = (self._get_video_tensor(delta := _next_maneuver_end - _next_maneuver_begin))
                assert len(frame_tensor) == 5+delta,"expected {} frames before prediction, got {}".format(5+delta,len(frame_tensor))
                frame_tensor = torch.stack([self.transform(x) for x in frame_tensor])  #apply to tensor
                

                label_tensor = torch.tensor(_manuever_type)
                
            except StopIteration as e:
                traceback.print_exc()

                raise 
            except AssertionError as e:
                frame_tensor , label_tensor = self.__next__
                return frame_tensor,label_tensor
            
            #switch channel - time segment dimensions ( 1,2)
            frame_tensor = frame_tensor.permute((1,0,2,3))
            assert   frame_tensor.size(0)==3 and frame_tensor.size(1)>0 and frame_tensor.size(2) == self.H and frame_tensor.size(3)==self.W,"got {} {} {} {}".format(frame_tensor.size(0),frame_tensor.size(1),frame_tensor.size(1),frame_tensor.size(1))

            
            return frame_tensor , label_tensor

    def _get_video_tensor(self , delta):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon + delta):
            #read 5 frames
            ret,val = self._cap.read()
            _ = next(self._current_timestep)
            if ret:
                _frames.append(val)
            else:
                logging.debug("Problem reading frame {}".format(j))
                # raise video_read_exception("Problem reading frame {}".format(j))
        return _frames
    
class read_frame_from_iter_val(base_class_prevention):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , 
                 
                 splits: List[float] =[0.8,0.1,0.1] ,
                 horizon: int = 5 ,
                 *args,
                 **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter_val).__init__()
        
        self.video_iter = self._cap_val

        self._current_timestep = iter(range(_val_start := self._max_train_frames*self._train))
        self._next_maneuver_begin = filter(self._next_maneuver_begin,lambda x : x>_val_start)
        self._next_maneuver_end = filter(self._next_maneuver_end,lambda x : x>_val_start)
        self._next_maneuver_end = filter(self._maneuver_type,lambda x : x>_val_start)

            
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
                    # raise ValueError()
            try:
                frame_tensor = (self._get_video_tensor(delta := _next_maneuver_end - _next_maneuver_begin))
                assert len(frame_tensor) == 5+delta,"expected {} frames before prediction, got {}".format(5+delta,len(frame_tensor))
                frame_tensor = torch.stack([self.transform(x) for x in frame_tensor])  #apply to tensor
                

                label_tensor = torch.tensor(_manuever_type)
                
            except StopIteration as e:
                traceback.print_exc()

                raise 
            except AssertionError as e:
                frame_tensor , label_tensor = self.__next__
                return frame_tensor,label_tensor
            
            #switch channel - time segment dimensions ( 1,2)
            frame_tensor = frame_tensor.permute((1,0,2,3))
            assert   frame_tensor.size(0)==3 and frame_tensor.size(1)>0 and frame_tensor.size(2) == self.H and frame_tensor.size(3)==self.W,"got {} {} {} {}".format(frame_tensor.size(0),frame_tensor.size(1),frame_tensor.size(1),frame_tensor.size(1))

            
            return frame_tensor , label_tensor

    def _get_video_tensor(self , delta):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon + delta):
            #read 5 frames
            ret,val = self._cap.read()
            _ = next(self._current_timestep)
            if ret:
                _frames.append(val)
            else:
                logging.debug("Problem reading frame {}".format(j))
                # raise video_read_exception("Problem reading frame {}".format(j))
        return _frames
    
class read_frame_from_iter_test(base_class_prevention):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , 
                 
                 splits: List[float] =[0.8,0.1,0.1] ,
                 horizon: int = 5 ,
                 *args,
                 **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter_test).__init__()
        
        self.video_iter = self._cap_test
        
        self._current_timestep = iter(range(_test_start := self._max_train_frames*self._test))
        self._next_maneuver_begin = filter(self._next_maneuver_begin,lambda x : x>_test_start)
        self._next_maneuver_end = filter(self._next_maneuver_end,lambda x : x>_test_start)
        self._next_maneuver_end = filter(self._maneuver_type,lambda x : x>_test_start)

            
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
                    # raise ValueError()
            try:
                frame_tensor = (self._get_video_tensor(delta := _next_maneuver_end - _next_maneuver_begin))
                assert len(frame_tensor) == 5+delta,"expected {} frames before prediction, got {}".format(5+delta,len(frame_tensor))
                frame_tensor = torch.stack([self.transform(x) for x in frame_tensor])  #apply to tensor
                

                label_tensor = torch.tensor(_manuever_type)
                
            except StopIteration as e:
                traceback.print_exc()

                raise 
            except AssertionError as e:
                frame_tensor , label_tensor = self.__next__
                return frame_tensor,label_tensor
            
            #switch channel - time segment dimensions ( 1,2)
            frame_tensor = frame_tensor.permute((1,0,2,3))
            assert   frame_tensor.size(0)==3 and frame_tensor.size(1)>0 and frame_tensor.size(2) == self.H and frame_tensor.size(3)==self.W,"got {} {} {} {}".format(frame_tensor.size(0),frame_tensor.size(1),frame_tensor.size(1),frame_tensor.size(1))

            
            return frame_tensor , label_tensor

    def _get_video_tensor(self , delta):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon + delta):
            #read 5 frames
            ret,val = self._cap.read()
            _ = next(self._current_timestep)
            if ret:
                _frames.append(val)
            else:
                logging.debug("Problem reading frame {}".format(j))
                # raise video_read_exception("Problem reading frame {}".format(j))
        return _frames

def main():
    pass

if __name__=="__main__":
    main()