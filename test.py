import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import tifffile as tif
import numpy as np

from network import Generator
from dataset_unsupervised import Synth_Dataset, Real_Dataset
from dataset_supervised import SAR_Dataset
from utils import test_Crop

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mu = 0.5
    gamma = 0.5
    batch_size = 16

    path2synth = '/home/minjun/Code/Data/SAR Data/DLSTR_RBOX_Pad/dlstr'
    path2real = '/home/minjun/Code/Data/SAR Data/DLSTR_RBOX_Pad/mstar'
    path2model = './Weight/DLSTR/CycleGAN/weights_RBOX_rot_pad_CycleGAN.pt'
    path2save = './Data/SAR Data/DLSTR/results/CycleGAN/RBOX_rot_pad'

    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Pad([0, 3, 0, 3], fill=0),
                                    # transforms.Resize([32, 32]),
                                    # transforms.CenterCrop(128),
                                    transforms.Normalize(mu, gamma)])
    
    test_ds = Synth_Dataset(path2synth, train = False, transform = transform)
    real_ds1 = Real_Dataset(path2real, train = False, transform = transform)
    real_ds2 = Real_Dataset(path2real, train = True, transform = transform)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    real_dl1 = DataLoader(real_ds1, batch_size = batch_size, shuffle = False)
    real_dl2 = DataLoader(real_ds2, batch_size = batch_size, shuffle = False)

    Gen_A2B = Generator().to(device)

    weight = torch.load(path2model)
    Gen_A2B.load_state_dict(weight)
    Gen_A2B.eval()
    to_pil = ToPILImage()
    with torch.no_grad():
        for synth, label, name in test_dl:
            fake_real = Gen_A2B(synth.to(device)).detach().cpu()

            for ii, img in enumerate(fake_real):
                path = os.path.join(path2save, 'refine', label[ii])
                os.makedirs(path, exist_ok = True)
                # img = img[:, 1:51, 1:51]
                img = to_pil(img * gamma + mu)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                #tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))

            for ii, img in enumerate(synth):
                path = os.path.join(path2save, 'synth', label[ii])
                os.makedirs(path, exist_ok = True)
                # img = img[:, 1:51, 1:51]
                img = to_pil(img * gamma + mu)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                # tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))

        for real, label, name in real_dl1:
            for ii, img in enumerate(real):
                path = os.path.join(path2save, 'real', label[ii])
                os.makedirs(path, exist_ok = True)
                # img = img[:, 1:51, 1:51]
                img = to_pil(img * gamma + mu)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                # tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))
        
        for real, label, name in real_dl2:
            for ii, img in enumerate(real):
                path = os.path.join(path2save, 'real', label[ii])
                os.makedirs(path, exist_ok = True)
                # img = img[:, 1:51, 1:51]
                img = to_pil(img * gamma + mu)
                img.save(os.path.join(path, name[ii][:-3]+'png'))

    
    # a
    # test_ds = SAR_Dataset('QPM', transform = transform, train = False)
    # test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    # with torch.no_grad():
    #     for synth, real, label, name in test_dl:
    #         fake_real = Gen_A2B(synth.to(device)).detach().cpu()

    #         for ii, img in enumerate(fake_real):
    #             path = os.path.join(path2save, 'refine', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))
            
    #         for ii, img in enumerate(synth):
    #             path = os.path.join(path2save, 'synth', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))

    #         for ii, img in enumerate(real):
    #             path = os.path.join(path2save, 'real', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))