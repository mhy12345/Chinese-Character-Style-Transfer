from data import SimpleDataset
from model import SmartModel
import argparse
import logging
import torch
import torch.utils.data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./dataset/image_100x100x64x64_stand.npy')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--sample_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
args = parser.parse_args()

dataset = SimpleDataset()
dataset.initialize(args)
data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        num_workers = 1)

model = SmartModel()

for i,(content_imgs, style_imgs, target) in enumerate(data_loader):
    pred = model(content_imgs, style_imgs)
