import cv2
import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")

# df = df[df["Target"]==1]

df["xmax"] = df["x"] + df["width"]
df["ymax"] = df["y"] + df["height"]
df_unique = df["patientId"].unique()

split_len = round(len(df_unique)*0.8)
train_unique = df_unique[0:split_len]
valid_unique = df_unique[split_len:]
train_df = df[df['patientId'].isin(train_unique)]
valid_df = df[df['patientId'].isin(valid_unique)]

transform = transforms.Compose([
    transforms.ToTensor()
])
class Opacity(Dataset):
    def __init__(self,dataframe,img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.df = dataframe
        self.img_ids = self.df["patientId"].unique()
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        info = self.df[self.df['patientId'] == img_id]
        img_1ch = pydicom.read_file("{}/{}.dcm".format(self.img_dir,img_id)).pixel_array
        img = np.stack([img_1ch]*3, -1)
        
        target = {}

        if 0 in np.array(info["Target"]):
            target["boxes"] = torch.zeros([0,4],dtype=torch.float32)
            target["labels"] = torch.zeros([0],dtype=torch.int64)
        else:
            boxes = torch.tensor(info[['x','y','xmax','ymax']].values,dtype=torch.float32)
            target["boxes"] = boxes
            target["labels"] = torch.ones((info.shape[0],),dtype=torch.int64)


        return transform(img),target
    def __len__(self):
        return len(self.img_ids)

train_set = Opacity(train_df,"/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images")
val_set = Opacity(valid_df,"/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images")
def collate_fn(batch):
    return tuple(zip(*batch))
train_loader = DataLoader(dataset=train_set,batch_size=4,collate_fn=collate_fn,num_workers=2)
val_loader = DataLoader(dataset=val_set,batch_size=4,collate_fn=collate_fn,num_workers=2)


from torchvision.models import EfficientNet_B3_Weights
from torchvision.models import efficientnet_b3
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import RetinaNet
backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT).features
backbone.out_channels = 1536
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
model = RetinaNet(backbone=backbone,num_classes=2,anchor_generator=anchor_generator)

# model = retinanet_resnet50_fpn_v2(num_classes=2,weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
model.to(device)


import time
from tqdm import tqdm
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = ReduceLROnPlateau(optimizer,factor=0.2,patience=4)
model.train()
itr = 1
total_train_loss = []
total_valid_loss = []
num_epochs = 1
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = []
    pbar = tqdm(train_loader, desc='let\'s train')
    for data in pbar:
        images,targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        losses_value = losses.item()
        train_loss.append(losses_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {losses_value}")
        itr += 1
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
    
    
    with torch.no_grad():
        valid_loss = []

        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            valid_loss.append(loss_value)
        epoch_valid_loss = np.mean(valid_loss)
        total_valid_loss.append(epoch_valid_loss)

    scheduler.step(epoch_valid_loss)
    print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
        f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")
        
test_img_dir = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images"
test_img_ids = os.listdir(test_img_dir)

test_result = pd.DataFrame(data=[],columns=["patientId","confidence","xmin","ymin","xmax","ymax"])

with torch.no_grad():
    for i,test_img_id in enumerate(test_img_ids):
        test_img = pydicom.read_file("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/{}".format(test_img_id)).pixel_array
        test_img = np.stack([test_img]*3, -1)
        test_img = torch.reshape(transform(test_img),[1,3,1024,1024]).to(device)
        model.eval()
        test_output = model(test_img)
        test_data = test_output[0]
        if test_data["boxes"].shape == torch.Size([0,4]):
            test_df = pd.DataFrame(
                [[test_img_id.split(".")[0],np.nan,np.nan,np.nan,np.nan,np.nan]],
                columns = ["patientId","confidence","xmin","ymin","xmax","ymax"]
            )
        else:
            test_df = pd.DataFrame(data = test_data["boxes"].cpu().numpy(),columns=["xmin","ymin","xmax","ymax"])

            test_df["confidence"] = pd.Series(data = test_data["scores"].cpu().numpy())
            test_df["patientId"] = test_img_id.split(".")[0]
        test_result = pd.concat([test_result,test_df],axis=0,ignore_index=True)
