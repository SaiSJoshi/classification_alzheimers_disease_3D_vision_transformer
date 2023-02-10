
import torch
from torch import nn
from torch.utils import data as torch_data
import wandb
import SimpleITK as sitk

from model import Model
LABEL = {0: 'AD', 1: 'CN'}
THRESHOLD = 0.5

IMG_PATH = "/content/gdrive/MyDrive/AD_NET/AD-Net/sai_stuff/vit/data/ad_test_ds/ADNI_130_S_0956_reg_brain.nii"
MODEL_PATH = "/content/gdrive/MyDrive/AD_NET/AD-Net/sai_stuff/vit/checkpointscheckpoint_2.pt"

model_path = MODEL_PATH if MODEL_PATH else input("Enter model path:  ")
img_path = IMG_PATH if IMG_PATH else input("Enter image path to classify:  ")

img = sitk.ReadImage(img_path)
np_img = sitk.GetArrayFromImage(img)
X = np_img.reshape(-1, 1, 128, 128, 64)

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()
output = model(X).squeeze(1)
output.reshape(-1)
if output[0] < THRESHOLD:
  print("Output label: ", LABEL[0])

else:
  print("Output label: ", LABEL[1])
