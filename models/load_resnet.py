import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet101
from torch.nn import functional as F
from ..conf.conf_py import _MODEL_
import logging
class resnet3d():
    def __init__(self) -> None:
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    def __call__(self):
        return self.model.forward()
    
class CNNLSTM(nn.Module):
    """
    CNN LSTM architecture custom
    """
    def __init__(self,  batch_size,num_classes=2,):
        """
        num_layers =num of cells

        """
        super(CNNLSTM, self).__init__()
        self.dev="cuda:0"
        self.batch_size=batch_size
        self.hidden_size = 1024 
        self.res_output_size = 300
        self.input_size = self.res_output_size+4 #(res features + 4 roi params)
        self.num_layers = 1
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Flatten(), 
                                       nn.Linear(self.resnet.fc.in_features, self.res_output_size)) #flatten feature, 300=inputsizelstm
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Linear(self.hidden_size,512) , nn.ReLU(), 
                                nn.Linear(512,256) , nn.ReLU() , 
                                nn.Linear(256 , 128), nn.ReLU() , 
                                nn.Linear(128,num_classes),nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
       
    def forward(self, x_3d, **roi_dict):
        self.h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.dev)  # hidden state
        self.c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.dev)  # internal state
        x_roi_centre,y_roi_centre=roi_dict["x_roi_centre"],roi_dict["y_roi_centre"]
        w_roi,h_roi =roi_dict["w_roi"] ,roi_dict["h_roi"]
        roi_param_tensor = torch.cuda.FloatTensor([x_roi_centre,y_roi_centre ,w_roi , h_roi]).unsqueeze(0)
        hidden=(self.h, self.c)
        cnn_features=[]
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  #conv features per time - flatten layer as mlp included ->produces output 300 features vector(batched)
                x = torch.cat((x,roi_param_tensor),dim=1)
                cnn_features.append(x)
                # cnn_features.extend([x_roi_centre,y_roi_centre,w_roi,h_roi])

        cnn_features_torch = torch.stack([x for x in cnn_features]) #stack temporal tensors(seq_len,batch,300)

        x = torch.permute(cnn_features_torch,(1,0,2)) #(batch,seq,f)
        # # x = torch.nn.Flatten(start_dim= 1,end_dim=-1)(x) #flatten conv features - (1,56,)
        # x= x.unsqueeze(0) #add batch dim
        logging.debug("Size of feature tensor input to lstm {}".format(x.size()))
        assert x.size(0)==1 and x.size(1)==56 and x.size(2)==304,"Expected input (1,56,304),got {}".format(x.size())

        out, hidden = self.lstm(x, hidden)   #input x is (batch,seq_len,features)=(1,x_3d.size(1),300)
        #out~(N,L,H)
        #hidden~(2,(num_layer,N,H)) from pytorch

        x = self.fc(out[:, -1, :]) #taking only last element for classification

        x= self.softmax(x)
        return x