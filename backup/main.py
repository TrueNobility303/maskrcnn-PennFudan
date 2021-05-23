import torch 
import torchvision 
from data import PennFudanDataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0)))        
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
model = FasterRCNN(backbone,num_classes=80,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)

myTransform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    transforms.Resize(300),
    transforms.CenterCrop(300)
])

dataset = CocoDetection('/datasets/Coco/train2017','/datasets/Coco/annotations/instances_train2017.json',transform=myTransform)
data = dataset[0]
x,y = data 
print(x.shape) 
print(y[0].keys())

dataloader = DataLoader(dataset,batch_size=32)
validset = CocoDetection('/datasets/Coco/val2017','/datasets/Coco/annotations/instances_val2017.json')
validloader = DataLoader(validset,batch_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_one_epoch(dataloader):
    for i,data in dataloader:
        x,y = data
        print(x.shape)
        print(y.shape)

if __name__ == '__main__':
    train_one_epoch(dataloader)


