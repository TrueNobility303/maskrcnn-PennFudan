import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead
import torchvision.transforms as T 
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm 
from config import device
from torch import nn
from maskrcnn import MaskRCNN_mobile_model
import matplotlib.pyplot as plt 

myTransform = T.Compose([
    T.ToTensor(),
    #T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    #T.Resize(300),
    #T.CenterCrop(300)
])

#将不定长元素直接转为元组形式
def my_collate_fn(batch):
    return tuple(zip(*batch))

class PennFudanDataset(object):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
       
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = myTransform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def train_one_epoch(model, optimizer, data_loader):
    model.train()
    tot_loss = 0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        #print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        tot_loss += losses.item()
    return tot_loss

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
#model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)

#使用预maskrcnn作为预训练的模型但更改roi_head和maskrcnn_head的num_classes
def get_instance_segmentation_model(num_classes,hidden_layer=256):
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model

num_classes = 2
#model = get_instance_segmentation_model(num_classes)
#print(model)
model = MaskRCNN_mobile_model
model.to(device)

dataset = PennFudanDataset('/datasets/PennFudan/PennFudanPed')
img, _ = dataset[0]
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
#print(prediction)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.005)

#使用collect_fn读取变长数据
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate_fn)  

def test_one_epoch(model,dataset,savepath):
    model.eval()
    img, _ = dataset[0]
    with torch.no_grad():
        prediction = model([img.to(device)])
    plt.figure()
    plt.clf()
    plt.subplot(1,3,1)
    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,3,2)
    pred_mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
    print(pred_mask.shape)
    plt.imshow(pred_mask,cmap='gray')
    plt.axis('off')

    plt.savefig(savepath)
    plt.close()
    
for epoch in range(100):
    loss = train_one_epoch(model,optimizer,data_loader)
    print('epoch',epoch,'loss',loss)
    test_one_epoch(model,dataset,'dump/'+str(epoch) + '.png')
    


