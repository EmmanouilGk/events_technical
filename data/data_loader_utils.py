import numpy as np
import torch

def gaussian_noise(img, mean=0, sigma=0.03):
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

def collate_fn_padding(batch):
    frames_batch , label_batch = zip(*batch)
    frames_batch=frames_batch[0] #untuple
    label_batch = label_batch[0]
   
    if (frames_batch.size(0))<=64:
        
            frames_batch_aug = torch.stack([gaussian_noise(frames_batch[-1] , mean=0 , sigma=0.05) for _ in range(64-len(frames_batch))])

            frames_batch = torch.cat([frames_batch_aug , frames_batch] , dim = 0)

    return frames_batch , label_batch