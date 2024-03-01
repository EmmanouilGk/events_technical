import numpy as np
import torch

from typing import List ,Tuple
from torch import FloatTensor

def gaussian_noise(img, mean=0, sigma=0.03):
    """
    add gaussian noise to images and pad frames with less than 64 frames

    
    """
    #3,600,1920
    #to tensor not rquiredd sinice we take statistcs (no need to 0-1)
    img= img.detach().cpu().numpy().transpose(1,2,0)  # chw hwc
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    img = torch.from_numpy(img)
    img = img.permute((2,0,1)) #hwc to chw
    img = img.unsqueeze(1) #add time dimension
    return img

def collate_fn_padding(batch:List[Tuple[FloatTensor,FloatTensor]])->Tuple[torch.FloatTensor , torch.FloatTensor]:
    """
    pad frame batches to maximum meanuever frames = 64
    padded_frames : how many start frames to insert to each
    """
    padded_frames:int = 30
    frames_batch , label_batch = zip(*batch)
    
    frames_batch=list(frames_batch)
    # label_batch = list(label_batch)
    for i, batch_item in enumerate(frames_batch):
         #batch item =(3,x,600,1920)

        if batch_item.size(1)<padded_frames:
            for _ in range(padded_frames - batch_item.size(1) ): 
                
                batch_item_aug = gaussian_noise(batch_item[: , 0 , : , :] ,mean=0,sigma=0.5)
                batch_item = torch.cat([ batch_item_aug ,  batch_item ] , dim=1 ) #insert last dims 

            assert batch_item.size(1)==padded_frames,"Expected 64 as 1st dim of iniput training tensor for video"
        else:
            batch_item = batch_item[: , :padded_frames , : ,: ]
        frames_batch[i]=batch_item   #process batc
         
    frames_batch = torch.stack([x for x in frames_batch])
    
    label_batch  = torch.stack([x for x in label_batch])

    assert frames_batch.size(1)==3 and frames_batch.size(2)==padded_frames

    return frames_batch , label_batch