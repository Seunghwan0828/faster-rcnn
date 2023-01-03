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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


warnings.filterwarnings('ignore')
SEED = 42

category_id = {1: "person"}

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

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        labels += 1

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
    A.Resize(300, 300)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("./weights/faster1/model30.pth"))


if __name__ == '__main__':
    testset = ConstDataset('data/valid/', 'data/annotations/valid_annotations.json', transforms)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, collate_fn=collate_fn)
    
    model.to('cuda')
    model.eval()

    image, target = next(iter(test_loader))
    outputs = model([image[0].to('cuda')])
    image = image[0].permute(1,2,0)
    image = image.numpy()

    # print(outputs[0])
    boxes = outputs[0]['boxes'].detach().cpu().numpy().tolist()
    scores = outputs[0]['scores'].detach().cpu().numpy().tolist()
    labels = outputs[0]['labels'].detach().cpu().numpy().tolist()
    print(outputs[0])

    cnt = 0
    for score in scores:
        if score > 0.3:
            cnt +=1
            
    del boxes[cnt:]
    del scores[cnt:]
    del labels[cnt:]

    for bbox,label in zip(boxes,labels):
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 1)
        cv2.putText(image, category_id[label], (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
    plt.imshow(image)
    plt.show()

                