import logging
from typing import Any, List, Tuple
from cv2 import VideoCapture
import numpy as np
import torch
from torchvision.transforms import ToTensor , Resize , Compose
import  cv2
import queue
import traceback
from abc import abstractmethod,ABC

from sklearn.preprocessing import OneHotEncoder

from copy import copy,deepcopy

from tqdm import tqdm
class video_read_exception(Exception):

    def __init__(self, message, ) -> None:
        super().__init__(message, )

def _return_prevention_class_encoded(val):
    if val == 3: 
        x=0
    if val == 4: 
        x=1
    if val == None: 
        x=2
    return x

class base_class_prevention(ABC):
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
        self._train , self._val , self._test = train ,val ,test
        self.path_to_video = path_to_video
        self.path_to_lane_changes = path_to_label
        self.H = 256
        self.W = 256
        self.horizon = prediction_horizon


        self.transform = Compose([ ToTensor() , Resize((self.H,self.W)) ]) #transfor for each read frame

        ###read and validate video
        self._cap_train = VideoCapture(self.path_to_video)
        self._cap_val=VideoCapture(self.path_to_video)
        self._cap_test=VideoCapture(self.path_to_video)

        # self._validate_video(self._cap_train , desc = "train"

        self._max_train_frames = int(self._cap_train.get(cv2.CAP_PROP_FRAME_COUNT))   #MAX_FRAMES - used for timestamp iterator
        input(self._max_train_frames)

        self.h , self.w = self._cap_train.get(cv2.CAP_PROP_FRAME_HEIGHT),self._cap_train.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.fps = self._cap_train.get(cv2.CAP_PROP_FPS)

        self._maneuver_type=2*np.ones((self._max_train_frames))

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
            self._maneuver_type[self._next_maneuver_end[-1]] = _return_prevention_class_encoded(_line[2])  #set maneuver type for this frame
            

            # self._next_step.put(labels[line+1][3])    #maneuver end times
        
        
        
    def __repr__(self):
        return "Training video sequence information\n FPS:{}\nMax Frames:{}\nSize{}x{}".format(self.fps,self._max_train_frames,
                                                                                                   self.h,self.w)



    def _validate_video(self,
                        video:cv2.VideoCapture,
                        desc:str)->None:
        """
        validate cv2 reads video frames with no error , for all length of video
        """
        with tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
            while video.isOpened():
                try:
                    ret,frame = video.read()
                    pbar.set_description_str("processing frame {}".format(idx:=video.get(cv2.CAP_PROP_POS_FRAMES)))
                    pbar.update(1)
                    if not ret and idx<video.get(cv2.CAP_PROP_FRAME_COUNT):
                        raise video_read_exception(message="video frame at position {} missing".format(idx))
                    if frame is None:
                        break
                except Exception as e:
                    raise e
                except KeyboardInterrupt as k:
                    break
        print("Video {} is ok".format(desc) )


class read_frame_from_iter_train(base_class_prevention,
                                 torch.utils.data.IterableDataset):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , *args, **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """        
        super(read_frame_from_iter_train,self).__init__(**kwargs)
        
        self._next_maneuver_begin = iter(self._next_maneuver_begin)
        self._next_maneuver_end = iter(self._next_maneuver_end)

        self._maneuver_type = iter(self._maneuver_type)
        self._current_timestep = iter(range( int(self._train*self._max_train_frames)))

        self._init_params = kwargs
   
    def __iter__(self):
        # while 1:
            self.video_iter = (self._cap_train)
            return self
    
    def _info_gen(self):
        """
        get current frame infor and update iterators
        """
        
        return next(self._current_timestep) , next(self._next_maneuver_begin) , next(self._next_maneuver_end) , next(self._maneuver_type)

    def _get_lane_keep_data(self, delta ):
        """
        return frame data stream when car keeps lane every self.hozizon time intervals
        """
        _frames_list=[]
        for _ in range(delta):
                _ = next(self._current_timestep) #update current time
                ret,_ = self._cap.read()  #move past redundant frames

                if not ret:
                    traceback.print_exc()
                _frames_list.append(ret)

                if len(_frames_list)==self.horizon:
                    _return_tensor= torch.stack([self.transform(x) for x in _frames_list])
                    yield _return_tensor , torch.tensor(_return_prevention_class_encoded(None))
                    _frames_list=[]
                    

                
    
    def __next__(self) -> Tuple[torch.FloatTensor , torch.FloatTensor]:
            
            """
            iterate over the video.
            Return video frame tensor (spatiotemporal) before maneuver (window frames)

            """
            if self.video_iter.isOpened():
            # _maneuver = _current_step[5]
            # _end = _current_step[6]
            # _maneuver_type=_current_step[3]

                _current_timestep , _next_maneuver_begin , _next_maneuver_end ,  _manuever_type = self._info_gen()

                #iter over redundant video frames
                if delta:=_current_timestep -_next_maneuver_begin - self.horizon>0 and _current_timestep + self.horizon < _next_maneuver_begin:
                    self._get_lane_keep_data( delta )

                try:
                    frame_tensor = (self._get_video_tensor(delta := _next_maneuver_end - _next_maneuver_begin))  #get 4d spatiotemporal tensor for pre-maneuver video sequence
                    
                    assert len(frame_tensor) == 5+delta,"expected {} frames before prediction, got {}".format(5+delta,len(frame_tensor))
                    
                    frame_tensor = torch.stack([self.transform(x) for x in frame_tensor])  #apply to tensor
                    

                    label_tensor = torch.tensor(_manuever_type)
                    
                except StopIteration as e:
                    traceback.print_exc()

                    raise 
                except AssertionError as e:
                    frame_tensor , label_tensor = self.__next__()
                    return frame_tensor,label_tensor
                
                #switch channel - time segment dimensions ( 1,2)
                frame_tensor = frame_tensor.permute((1,0,2,3))
                assert   frame_tensor.size(0)==3 and frame_tensor.size(1)>0 and frame_tensor.size(2) == self.H and frame_tensor.size(3)==self.W,"got {} {} {} {}".format(frame_tensor.size(0),frame_tensor.size(1),frame_tensor.size(1),frame_tensor.size(1))

                
                return frame_tensor , label_tensor
            raise StopIteration
    
    def _restart(self):
        """
        restart iterators for multi-epoch batch training
        """
        super(read_frame_from_iter_train , self).__init__(self._init_params)


    def _get_video_tensor(self , delta):
        """
        get one video sequence for a manuever method. Return 5 frames 
        """
        _frames = []
        for j in range(self.horizon + delta):
            #read 5 frames
            ret,val = self.video_iter.read()
            _ = next(self._current_timestep)
            if ret:
                _frames.append(val)
            else:
                logging.debug("Problem reading frame {}".format(j))
                raise video_read_exception("Problem reading frame {} of video frame {}".format(j , self.video_iter.get(cv2.CAP_PROP_POS_FRAMES)))
        return _frames
    
class read_frame_from_iter_val(base_class_prevention,
                               torch.utils.data.IterableDataset):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , 
                
                 *args,
                 **kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter_val,self).__init__(**kwargs)

        self._current_timestep = iter(range(_val_start := int(self._max_train_frames*self._train) , int(self._max_train_frames*self._val)))
        self._next_maneuver_begin = filter(lambda x : x>_val_start , self._next_maneuver_begin, )
        self._next_maneuver_end = filter(lambda x : x>_val_start , self._next_maneuver_end,)
        self._next_maneuver_type =  iter(self._maneuver_type[_val_start:])

        input(next(self._next_maneuver_type))

            
    def __iter__(self):
        self.video_iter = (self._cap_val)
        self._cap_val.set(cv2.CAP_PROP_POS_FRAMES, self._val*self._max_train_frames)

        return self
    
    def _info_gen(self):
        """
        get current frame infor and update iterators
        """
        
        return next(self._current_timestep) , next(self._next_maneuver_begin) , next(self._next_maneuver_end) , next(self._maneuver_type)
    
    def _get_lane_keep_data(self, delta ):
        """
        return frame data stream when car keeps lane every self.hozizon time intervals
        """
        _frames_list=[]
        for _ in range(delta):
                _ = next(self._current_timestep) #update current time
                ret,_ = self._cap.read()  #move past redundant frames

                if not ret:
                    traceback.print_exc()
                _frames_list.append(ret)

                if len(_frames_list)==self.horizon:
                    _return_tensor= torch.stach([self.transform(x) for x in _frames_list])
                    yield _return_tensor , torch.tensor(_return_prevention_class_encoded(None))

                
    
    def __next__(self) -> Tuple[torch.FloatTensor , torch.tensor]:
            """
            iterate over the video.
            Return video frame tensor (spatiotemporal) before maneuver (window frames)

            """

            # _maneuver = _current_step[5]
            # _end = _current_step[6]
            # _maneuver_type=_current_step[3]

            _current_timestep , _next_maneuver_begin , _next_maneuver_end ,  _manuever_type = self._info_gen()
            input(_current_timestep)
            #iter over redundant video frames
            if delta:=_current_timestep -_next_maneuver_begin - self.horizon>0:
                self._get_lane_keep_data( delta )
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
    
class read_frame_from_iter_test(base_class_prevention,
                                torch.utils.data.IterableDataset):
    """
    Torch iterable dataset impolementation for reading 4D camera input for manuever 
    detection , training set.

    path_to_video: path to prevention video - raw camera leftImg
    path_to_label: full video duration labels
    horizon: prediction window
    """


    def __init__(self , *args,**kwargs) -> None:
        """
        attributes
        path_to_lane_changes: lane_changes.txt
        cap: video capture for video path
        k : increment for constructing 4D image dataset
        """
        super(read_frame_from_iter_test,self).__init__(**kwargs)
        
        self.video_iter = self._cap_test
        
        self._current_timestep = iter(range(_test_start := int(self._max_train_frames*self._test),self._max_train_frames))
        self._next_maneuver_begin = filter(lambda x : x>_test_start,self._next_maneuver_begin)
        self._next_maneuver_end = filter(lambda x : x>_test_start,self._next_maneuver_end)
        self._next_maneuver_end = filter(lambda x : x>_test_start,self._maneuver_type)

            
    def __iter__(self):
        self.video_iter = self._cap_test
        self._cap_test.set(cv2.CAP_PROP_POS_FRAMES, self._test*self._max_train_frames)

        return self
    
    def _info_gen(self,verbose):
        """
        get current frame infor and update iterators
        """
        if verbose:
            print("Current time step in video read is {}".format(ct := next(self._current_timestep)))
            print("Next maneuver begin step in video read is {}".format(mb := next(self._next_maneuver_begin)))
            print("Next maenuber end step in video read is {}".format(me := next(self._next_maneuver_end)))
            print("Maneuver type in video read is {}".format(mt := next(self._maneuver_type)))

        return ct,mb,me,mt
    
    def __next__(self) -> Tuple[torch.FloatTensor , torch.FloatTensor]:
            """
            iterate over the video.
            Return video frame tensor (spatiotemporal) before maneuver (window frames)

            """

            # _maneuver = _current_step[5]
            # _end = _current_step[6]
            # _maneuver_type=_current_step[3]

            _current_timestep , _next_maneuver_begin , _next_maneuver_end ,  _manuever_type = self._info_gen(verbose=True)
            
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