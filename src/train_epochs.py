import os
import contextlib
from typing import Dict, Optional
import traceback
from detectron2.data.catalog import MetadataCatalog
from detectron2.structures import Boxes
import numpy as np
import torch
from torch.torch_version import TorchVersion
from torchvision.transforms import Resize
from detectron2.config import get_cfg
from .train_utils import MyCustomError
from intention_prediction.src.train_utils import apply_bboxes
from intention_prediction.src.train_utils import apply_bboxes_single
from ..conf.conf_py import _PADDED_FRAMES
import cv2
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
import torchvision
from detectron2.utils.visualizer import Visualizer, ColorMode


from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score,recall_score
from ..data.data_loader_utils import collate_fn_padding

from ..data.prevention_data_iter import read_frame_from_iter_train, read_frame_from_iter_val
class logging_utils():
    """
    logging helper methods for ML models info and debug
    """
    

    def __init__(self,
                 writer_path,
                 len_dataloader,
                 current_epoch) -> None:
      
        self._current_epoch = current_epoch


    def tb_writer(self, func):
        def wrapper(*args,**kwargs):
            losses = func(*args ,**kwargs)
            self.writer.addScalar(losses["desc"] , losses["val"] , losses["time"])
        return wrapper
    
    @property
    def epoch(self):
        print("Getting value...")
        return self._current_epoch

    @epoch.setter
    def epoch(self, value):
        print("Setting value...")
        self._current_epoch = value


def _write_val_values(val_losses_dict:dict,
                      losses_dict:dict
                      , writer:SummaryWriter , 
                      epoch:int , 
                       )->None:
      """
      
      write train/val dict metrics in tb

      """
      for i, (desc , val) in enumerate(val_losses_dict.items()):
            for set in ["val","train"]:
                if set=="val":batches=val_losses_dict["batch_count"]
                elif set=="train":batches=losses_dict["batch_count"]
                if desc=="loss_{}_epoch".format(set): 
                    for i in range(val.shape[0]):
                        writer.add_scalar("{} Batch_Loss".format(set) , val[i] , (epoch-1)*batches + i)  #plot val loss
                if desc=="{}_pres".format(set):
                    for i in range(2):
                        writer.add_scalars("{} micro (0=Lk,1=Llc,2=Rlc)".format(set) , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                    writer.add_scalar("{} Macro".format(set) , val[0],  epoch)   # add epoch wise acc,pres etc metrics 

                    continue
                if desc =="{}_pres_weighted".format(set):
                    for i in range(2):
                        print(val)
                        writer.add_scalars("{}_pres_weighted".format(set) , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                    continue
                if desc=="{}_rec".format(set):
                    for i in range(2):
                        writer.add_scalars("Rec micro" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 

                    writer.add_scalar("Rec macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 
                    continue
            
                if desc=="{}_acc".format(set):
                    writer.add_scalar(desc , val,  epoch)   # add epoch wise acc,pres etc metrics 
                if desc=="{}_bacc".format(set):
                    writer.add_scalar(desc , val,  epoch)   # add epoch wise acc,pres etc metrics 

def save_model(epoch:int,
               model:torch.nn.Module,
               optimizer:torch.optim,
               dstp:str, **kwargs)->None:
    
    if "loss" in kwargs:
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "loss" : kwargs["losses_dict"]["desc"][-1],
                }, dstp)
    else:
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                }, dstp)

def train(*args,**kwargs):
    """
    config batch training and validtion
    expexted kwargs:
     writer,scheduler,model,optimizer,dataloader (train and val), device, save path model weights
    """
    max_epochs:int = kwargs["epochs"]  #max train epoch
    dev = kwargs["dev"]
    now = datetime.datetime.now()
    scheduler: torch.optim.lr_scheduler.ExponentialLR = kwargs["scheduler"]


    # #load weights for training
    if "weights" in kwargs:
        weights=kwargs["weights"]
        criterion = torch.nn.CrossEntropyLoss( weight= torch.tensor(data = ( weights[0] , weights[1], weights[2]), dtype=torch.float , device=dev))
        
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # print(weights)
    # weights = [weights["LK"],weights["LLC"],weights["RLC"],]
    
    model=kwargs["model"]
    
 
    ##update config params
    kwargs.update({"criterion":criterion})
    class_map=dict({0:"LK",
                    1:"LLC",
                    2:"RLC"})

   
    writer= kwargs["writer"]
    
    torch.backends.cudnn.benchmark = True

    optimizer=kwargs["optimizer"]
    current_epoch=0
    if "load_saved_model" in kwargs:
        print("Loading saved model at path {}".format(kwargs["load_saved_model"]))
        checkpoint = torch.load(kwargs["load_saved_model"])
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model=model.to(dev)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        
    patience=0
    #train and val
    for epoch in  range(0 , max_epochs-current_epoch):

        #train and val
        losses_dict = train_one_epoch(*args , **kwargs)   #train losses dict

        _write_val_values(val_losses_dict= losses_dict ,
                          writer=writer,
                          epoch=epoch,
                          losses_dict=losses_dict)
        
        save_model(epoch,
                   model,
                   optimizer,
                   kwargs["model_save_path"] , 
                   losses_dict = losses_dict)

        val_losses_dict = val_one_epoch_given_obj_detection(*args, **kwargs)  #vla losses dict
        
        _write_val_values(val_losses_dict=val_losses_dict , 
                          losses_dict= losses_dict,
                          writer=writer,
                          epoch=epoch,
                          )

        #early stoppping - learing rate adapation
        if (val_losses_dict["loss_val_epoch"][-1] - losses_dict["loss_train_epoch"][-1])>0:
            patience+=1
        if patience==3:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"]/10
        if patience==6:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"]/10
        if patience == 10: 
            save_model(epoch,model,optimizer,kwargs["model_save_path"])
            raise Exception("Early Stopping triggered at epoch {}\nStopping training".format(epoch))


        scheduler.step()

        # writer.add_hparams({"lr" : scheduler.get_last_lr() } ,
        #                     {"loss_mean_val":np.mean(val_losses_dict["loss_val_epoch"])})

        save_model(epoch,
                   model,
                   optimizer ,
                   kwargs["model_save_path"])

def train_one_epoch(*args , **kwargs):
        """
        train batches logic
        """
        data_loader = kwargs["dataloader_train"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        writer= kwargs["writer"]
        loss_epoch = []
        
        accumulated_gradients = kwargs["num_iterations_gr_accum"]

        max_batches = 0
        predictions_epoch=[]
        labels_epoch=[]
        
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 

            pbar.set_description_str("Train Batch: {}".format(batch_idx))
            
            frames = frames.to(dev)
            maneuver_type=maneuver_type.to(dev)

            prediction = model(frames)

            loss = torch.nn.functional.cross_entropy(prediction,
                                                     maneuver_type , 
                                                     weight= torch.tensor(data = ( 20,32), 
                                                     dtype=torch.float , 
                                                     device=dev))
            print(maneuver_type)
            print(loss)
            loss=loss/accumulated_gradients
            # loss = criterion(prediction , maneuver_type) / accumulated_gradients

            loss.backward()
            loss_epoch.append(loss.item())
            writer.add_scalar("Online batch loss",loss_epoch[-1],batch_idx)
            
            if ((batch_idx + 1) % accumulated_gradients == 0):
                optimizer.step()
                optimizer.zero_grad()
                max_batches+=1
                writer.add_scalar("Online batch loss-During update ", loss_epoch[-1] , batch_idx)

          
            pbar.set_postfix_str("Batch loss {:0.2f}".format(loss.item()))

            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())
            

        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=list(map(lambda x: int(x) , labels_epoch))

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        bacc = balanced_accuracy_score(labels_epoch , predictions_epoch)

        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = "micro")
        pres=precision_score(labels_epoch , predictions_epoch , average = "weighted")

        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= "micro")

        return {"loss_train_epoch":np.array(loss_epoch),
                "train_acc":acc,
                "trian_pres":[pres_avg , pres_class] , 
                "train_rec":[rec,rec_class] ,
                "train_pres_global":pres,
                "batch_count":len(data_loader),
                "train_bacc":bacc}

@torch.no_grad
def val_one_epoch(*args , **kwargs)->Dict:
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0
        
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 
            pbar.set_description_str("Val Batch: {}".format(batch_idx))
            
            frames = frames.to(dev) 
            maneuver_type=maneuver_type.type(torch.LongTensor).to(dev)
            

            prediction = model(frames)
           
            loss = criterion(prediction, maneuver_type)

            #to comute eval metrics
            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            loss_epoch.append(loss.item())
       
            pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))

            max_epochs_val+=1



        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=list(map(lambda x: int(x) , labels_epoch))

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        bacc = balanced_accuracy_score(labels_epoch , predictions_epoch)

        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = "micro")
        pres=precision_score(labels_epoch , predictions_epoch , average = "weighted")

        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= "micro")

        print(pres_class)
        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":[pres_avg , pres_class] , 
                "val_rec":[rec,rec_class] ,
                "batch_count":max_epochs_val , 
                "val_pres_global":pres,
                "val_bacc":bacc}

@torch.no_grad
def val_one_epoch_given_obj_detection(*args , **kwargs)->Dict:
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0
        
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 
            pbar.set_description_str("Val Batch: {}".format(batch_idx))
            
            frames = frames.to(dev) 
            maneuver_type=maneuver_type.type(torch.LongTensor).to(dev)

            prediction = model(frames)
           
            loss = criterion(prediction, maneuver_type)

            #to comute eval metrics
            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            loss_epoch.append(loss.item())
       
            pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))

            max_epochs_val+=1



        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=list(map(lambda x: int(x) , labels_epoch))

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        bacc = balanced_accuracy_score(labels_epoch , predictions_epoch)

        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = "micro")
        pres=precision_score(labels_epoch , predictions_epoch , average = "weighted")

        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= "micro")

        print(pres_class)
        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":[pres_avg , pres_class] , 
                "val_rec":[rec,rec_class] ,
                "batch_count":max_epochs_val , 
                "val_pres_global":pres,
                "val_bacc":bacc}


@torch.no_grad
def val_one_epoch_with_single_detection(*args , **kwargs)->Dict:
        """
        validate one epoch
        """
        data_loader = kwargs["dataloader_val"]
        dev=kwargs["dev"]
        model = kwargs["model"].to(dev)
        criterion = kwargs["criterion"]
        detector: DefaultPredictor = kwargs["detector"]

        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0
        
        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 
            pbar.set_description_str("Val Batch: {}".format(batch_idx))
            
            frames:torch.tensor = frames.to(dev) 
            maneuver_type:torch.tensor =maneuver_type.type(torch.LongTensor).to(dev)

            
            frames = frames.squeeze(0) #remove batch dim
            frames = torch.permute(frames , (1,0,2,3))  #1st segment dimension

            assert frames.dim()==4 and frames.size(1)==3  
            
            frames = frames.detach().cpu().numpy() #to numpy
            
            #find bboxes by detectron2
            bboxes = []
            pred_classes = []
            conf=[]
            for i,frame in enumerate(frames):
                
                frame = np.transpose(frame , (1,2,0)) *255  #hwc and uint8
                
                assert frame.shape[-1]==3


                try:
                    box1 = detector(frame)["instances"].pred_boxes
                    

                    if box1.tensor.numel()!=0:
                        bboxes.append(box := box1[0])  ##keep only 1st detection
                        pred_classes.append(detector(frame)["instances"].pred_classes[0])
                        conf.append(detector(frame)["instances"].scores[0])
                    else:
                        bboxes.append(Boxes(torch.tensor([[0,frame.shape[0] , 0,frame.shape[1]]] , dtype=torch.float , device=dev)))
                        pred_classes.append(None)
                        conf.append(None)

                except (IndexError,AssertionError) as e:
                    print("Empty bbox for frame {i}".format(i=i  ))
                    traceback.print_exc()
                    raise 
                    
                
                v = Visualizer(frame)
                out = v.draw_instance_predictions(detector(frame)["instances"].to("cpu"))

                cv2.imwrite(filename=os.path.join("/home/iccs/Desktop/isense/events/intention_prediction/debug_2/","test_image{}.png".format(i)) ,
                            img = frame[:,:,::-1])
                cv2.imwrite(filename=os.path.join("/home/iccs/Desktop/isense/events/intention_prediction/debug/","test_img_with_detections_{}.png".format(i)) ,
                            img=out.get_image()[:,:,::-1], )
                


                print("Bbox {} found empty{}".format(bboxes[-1] , bboxes[-1].tensor.numel()))

            ##apply bboxes to frames of segment.
            try:
                assert len(frames)>0

                frames_cropped = apply_bboxes_single(frames , bboxes ,  pred_classes , conf)  #return MxN frames,where n= frames in segment,M=Detections

                for j,i in enumerate(frames_cropped):
                    out = v.draw_instance_predictions(detector(i.transpose(1,2,0))["instances"].to("cpu"))

                    cv2.imwrite(filename=os.path.join("/home/iccs/Desktop/isense/events/intention_prediction/debug/","test_img_with_detections_{}_after_crop.png".format(i)) ,
                            img=out.get_image().transpose(1,2,0), )
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
      

            #loop over frame, detection is singular
            for frame in frames_cropped:
                
                    frame = torch.from_numpy(frame)
                    frame = frame.to(dev)
                    
                    prediction = model(frame)
            
                    loss = criterion(prediction, maneuver_type)

                    #to comute eval metrics
                    predictions_epoch.append(prediction.detach().cpu().numpy())
                    labels_epoch.append(maneuver_type.detach().cpu().numpy())

                    loss_epoch.append(loss.item())
            
                    pbar.set_postfix_str("Val Batch loss {:0.2f}".format(loss.item()))



        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=     list(map(lambda x: int(x) , labels_epoch))

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
        bacc = balanced_accuracy_score(labels_epoch , predictions_epoch)

        pres_avg=precision_score(labels_epoch , predictions_epoch , average = "macro")
        pres_class=precision_score(labels_epoch , predictions_epoch , average = "micro")
        pres=precision_score(labels_epoch , predictions_epoch , average = "weighted")

        rec =recall_score(labels_epoch , predictions_epoch , average="macro")
        rec_class = recall_score(labels_epoch , predictions_epoch , average= "micro")

        print(pres_class)
        return {"loss_val_epoch":np.array(loss_epoch),
                "val_acc":acc,
                "val_pres":[pres_avg , pres_class] , 
                "val_rec":[rec,rec_class] ,
                "batch_count":len(data_loader) , 
                "val_pres_global":pres,}