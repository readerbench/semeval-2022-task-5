import os, sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
import torchvision.transforms as t

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from alexnet import KitModel as AlexNet
from vgg19 import KitModel as VGG19


class ImageListDataset (Dataset):

    def __init__(self, image_folder, transform=None):
        super(ImageListDataset).__init__()
    
        
        self.list = list(Path(image_folder).glob('*'))
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.list[index]
            
        x = default_loader(path)
        if self.transform:
            x = self.transform(x)
        
        return x, path.name
    
    def __len__(self):
        return len(self.list)

    
def main(args):
    assert Path(args.model_file).exists(), f"Model file {args.model_file} does not exist"
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Lambda(lambda x: x[[2,1,0], ...] * 255),  # RGB -> BGR and [0,1] -> [0,255]
        t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1,1,1]),  # mean subtraction
    ])

    data = ImageListDataset(args.image_folder,  transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    model = AlexNet if 'hybrid' in args.model_file else VGG19
    model = model(args.model_file).to('cuda')
    # if getattr(model, 'conv5_4'):
    #     model.conv5_4.register_forward_hook(get_activation('conv5_4'))
    # else:
    #     model.conv5.register_forward_hook(get_activation('conv5'))
    
    model.fc7_1.register_forward_hook(get_activation('fc7_1'))

    model.eval()
    
    with torch.no_grad():
        for x, filenames in tqdm(dataloader):
            activation = {}
            p = model(x.to('cuda')).cpu().numpy()  # order is (NEG, NEU, POS)
            # features = activation['conv5_4']
            
            # if getattr(model, 'conv5_4'):
            #     relu5_4         = F.relu(features)
            #     pool5_pad       = F.pad(relu5_4, (0, 1, 0, 1), value=float('-inf'))
            #     pool5           = F.max_pool2d(pool5_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
            # else:
            #     relu5           = F.relu(features)
            #     pool5_pad       = F.pad(relu5, (0, 1, 0, 1), value=float('-inf'))
            #     pool5           = F.max_pool2d(pool5_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)

            features = activation['fc7_1']
            
            features         = F.relu(features)


            for idx, file in enumerate(filenames):
                np.save(open(out_dir / Path(file).with_suffix('.npy'),"wb"), features[idx].cpu().numpy())

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict Visual Sentiment')
    parser.add_argument('--image-folder', type=str, help='Image list (txt, one path per line)')
    parser.add_argument('-m', '--model-file', type=str,  default='vgg19_finetuned_all.pth', help='Pretrained model filepath')
    parser.add_argument('-w', '--num-workers', type=int, default=8, help='Num Workers')
    parser.add_argument('-b', '--batch-size', type=int, default=48, help='Batch size')
    parser.add_argument('-o', '--output-folder', type=str, help='Feature save folder')
    args = parser.parse_args()
    main(args)
