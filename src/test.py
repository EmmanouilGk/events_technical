from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score,balanced_accuracy_score,f1_score
import torch
from torchvision.utils import np
from tqdm import tqdm


@torch.no_grad
def test(*args , **kwargs)->Dict:
        """
        test
        """
        writer = kwargs["writer"]
        val_losses_dict = test_batches(**kwargs)
        epoch=0
        for i, (desc , val) in enumerate(val_losses_dict.items()):
          
            if desc=="val_pres":
                for i in range(2):
                    writer.add_scalars("Val micro (0=Lk,1=Llc,2=Rlc)" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                writer.add_scalar("Val Macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 

                continue
            if desc =="val_pres_weighted":
                for i in range(2):
                    print(val)
                    writer.add_scalars("val_pres_weighted" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 
                continue
            if desc=="val_rec":
                for i in range(2):
                    writer.add_scalars("Rec micro" , {"Class {}".format(i):val[1]},  i)   # add epoch wise acc,pres etc metrics 

                writer.add_scalar("Rec macro" , val[0],  epoch)   # add epoch wise acc,pres etc metrics 
                continue
        
            if desc=="acc":
                writer.add_scalar(desc , val,  epoch)   # add epoch wise acc,pres etc metrics 
                              

def test_batches(*args , **kwargs):
        data_loader = kwargs["dataloader_test"]
        dev=kwargs["dev"]
        model = kwargs["model"]
        
        loss_epoch = []
        predictions_epoch = []
        labels_epoch = []
        max_epochs_val = 0

        print("Loading saved model at path {}".format(kwargs["load_saved_model"]))
        checkpoint = torch.load(kwargs["load_saved_model"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model=model.to(dev)

        for batch_idx , (frames , maneuver_type) in (pbar:=tqdm(enumerate(data_loader))): 
            pbar.set_description_str("Val Batch: {}".format(batch_idx))
            
            frames[0],frames[1]=frames[0].to(dev),frames[1].to(dev)
            maneuver_type=maneuver_type.type(torch.LongTensor).to(dev)
            

            prediction = model(frames)
           

            #to comute eval metrics
            predictions_epoch.append(prediction.detach().cpu().numpy())
            labels_epoch.append(maneuver_type.detach().cpu().numpy())

            last_predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
            last_labels_epoch=list(map(lambda x: int(x) , labels_epoch))

            print(last_predictions_epoch)
            print(last_labels_epoch)
            pbar.set_postfix_str("Running Accuracy {:0.2f}".format(accuracy_score(last_labels_epoch , last_predictions_epoch)))
            print("Running Accuracy {:0.2f}".format(accuracy_score(last_labels_epoch , last_predictions_epoch)))
            print("Running Pr {:0.2f}".format(precision_score(last_labels_epoch , last_predictions_epoch,average="weighted")))
            print("Running Bacc {:0.2f}".format(balanced_accuracy_score(last_labels_epoch , last_predictions_epoch)))
            print("Running F1 {:0.2f}".format(f1_score(last_labels_epoch , last_predictions_epoch,average="weighted")))



            max_epochs_val+=1

        #convert to int categorical labels
        predictions_epoch=list(map(lambda x: np.argmax(x) , predictions_epoch))
        labels_epoch=list(map(lambda x: int(x) , labels_epoch))

        
        acc = accuracy_score(labels_epoch , predictions_epoch)
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
                "val_pres_global":pres}

def test_conf(*args , **kwargs):
      writer = kwargs["writer"]
      test_dict = test(args,kwargs)
      