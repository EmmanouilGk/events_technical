import torch
class resnet3d():
    def __init__(self) -> None:
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    def __call__(self):
        return self.model.forward()