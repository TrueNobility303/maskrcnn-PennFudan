import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
      
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7,sampling_ratio=2)
mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14,sampling_ratio=2)
MaskRCNN_mobile_model = MaskRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,mask_roi_pool=mask_roi_pooler)

if __name__  == '__main__':
    model = MaskRCNN_mobile_model
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)