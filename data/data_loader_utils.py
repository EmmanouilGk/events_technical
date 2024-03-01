import numpy as np
import torch

from typing import List ,Tuple
from torch import FloatTensor

def gaussian_noise(img, mean=0, sigma=0.03):
    """
    add gaussian noise to images and pad frames with less than 64 frames
    """
    img= img.detach().cpu().numpy().transpose(1,2,0)
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    img = torch.from_numpy(img)
    img = img.permute((2,0,1))
    return img

def collate_fn_padding(batch:List[Tuple[FloatTensor,FloatTensor]])->Tuple[torch.FloatTensor , torch.FloatTensor]:
    """
    pad frame batches to maximum meanuever frames = 64
    """
    frames_batch , label_batch = zip(*batch)
    
    for i, batch_item in enumerate(frames_batch):
         for _ in range(64 - batch_item.size(1) ): 
              batch_item = torch.cat([gaussian_noise(batch_item[: , -1 , : , :] ,mean=0,sigma=0.5) ,  batch_item ] , dim=1 ) #insert last dims 

         assert batch_item.size(1)==64,"Expected 64 as 1st dim of iniput training tensor for video"
         frames_batch[i]=batch_item   #process batch

    # if (frames_batch.size(0))<=64:
        
    #         frames_batch_aug = torch.stack([gaussian_noise(frames_batch[-1] , mean=0 , sigma=0.05) for _ in range(64-len(frames_batch))])

    #         frames_batch = torch.cat([frames_batch_aug , frames_batch] , dim = 0)

    return frames_batch , label_batch