import os
import warnings
from typing import Tuple, Sequence, Callable
import json
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.optim as optim
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.vgg import VGG16_Weights, vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import wandb

warnings.filterwarnings('ignore')

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


class ConstDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_path: os.PathLike,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        self.label_path = label_path
        self.transforms = transforms

        with open(self.label_path, 'r') as f:
            annots = json.load(f)
        
        self.annots = annots


    def __len__(self) -> int:
        return len(self.annots['images'])
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.annots['images'][index]
        file_name = os.path.join(self.image_dir, image_id['file_name'])
        image = Image.open(file_name).convert('RGB')
        image = np.array(image)

        annots = [x for x in self.annots['annotations'] if x['image_id'] == image_id['id']]
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int64)


        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            transformed_img = transformed['image']
            transformed_bbox = transformed['bboxes']
            transformed_label = transformed['class_labels']


        transformed_img = transformed_img.transpose(2,0,1) # hwc to hwc
        transformed_img = transformed_img / 255.0  # 0-1
        transformed_img = torch.Tensor(transformed_img)


        target = {
            'boxes': transformed_bbox,
            'labels': transformed_label
        }

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        return transformed_img, target


def collate_fn(batch):
    return tuple(zip(*batch))

transforms = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


epochs = 100
optimizer = optim.Adam(model.parameters(), lr=1e-4)


if __name__ == '__main__':
    trainset = ConstDataset('data/train/', 'data/annotations/train_annotations.json', transforms)

    train_loader = DataLoader(trainset, batch_size=16, num_workers=0, collate_fn=collate_fn)

    model.to('cuda')
    model.train()
    for epoch in range(1, epochs+1):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            images = [image.to('cuda') for image in images]
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            wandb.log({
                "loss": losses,
                "loss_classifier": loss_dict["loss_classifier"],
                "loss_box_reg": loss_dict["loss_box_reg"],
                "loss_objectness": loss_dict["loss_objectness"]
            })

            losses.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch {epoch} - Total: {losses:.4f}, loss_classifier: {loss_dict["loss_classifier"]:.4f}, loss_box_reg: {loss_dict["loss_box_reg"]:.4f}, loss_objectness: {loss_dict["loss_objectness"]:.4f}')
        torch.save(model.state_dict(), f'./weights/model{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses
        }, f'./weights/resume_model{epoch}.pth')
