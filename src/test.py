from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from torchvision.utils import np

from tqdm import tqdm


@torch.no_grad
def test(*args , **kwargs)->Dict:
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"]
        criterion = kwargs["criterion"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 

            pbar.set_description_str("Batch: {}/{}".format(batch_idx))
            
            frames = frames.to(dev)
            maneuver_type=maneuver_type.to(dev)
            prediction = model(frames)
            prediction =torch.nn.Softmax(prediction)
           
            loss = criterion(prediction , maneuver_type)

            #to comute eval metrics
            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            loss_epoch.append(loss.item())

            pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))

            max_epochs_val+=1

        acc = accuracy_score(labels_epoch , predictions_epoch)
        pres=precision_score(labels_epoch , predictions_epoch)
        rec =recall_score(labels_epoch , predictions_epoch)

        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":pres , 
                "val_rec":rec ,
                "batch_count":max_epochs_val}