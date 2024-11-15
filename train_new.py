
from dataset_unsupervised import Real_Dataset, Synth_Dataset
from network import Generator, Discriminator
from utils import Crop, Rotation, ReplayBuffer, test_Crop, Angle_Rotation, Add_Noise

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import os
import time
import itertools
import matplotlib.pyplot as plt

def plot(synth, real, label, mu, gamma):
    plt.figure(figsize = [10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(synth.squeeze() * gamma + mu, cmap = 'gray')
    plt.axis('off')
    plt.title('Synth/' + label)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(real.squeeze() * gamma + mu, cmap = 'gray')
    plt.axis('off')
    plt.title('Real/' + label)
    plt.colorbar()
    plt.show()
    plt.savefig('aa.png')

def initialize_weight(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

if __name__ == '__main__':

    # Hyper Parameter
    mu = 0.5
    gamma = 0.5
    batch_size = 16
    lr = 1e-4
    num_epoch = 300

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mu, gamma)])

    synth_path = '/home/minjun/Code/Data/SAR Data/DLSTR_RBOX_Pad/dlstr'
    real_path = '/home/minjun/Code/Data/SAR Data/DLSTR_RBOX_Pad/mstar'

    real_ds = Real_Dataset(real_path, transform = transform, train = True)
    synth_ds = Synth_Dataset(synth_path, transform = transform, train = True)

    path = '/home/minjun/Code/Weight/DLSTR/CycleGAN'
    writer = SummaryWriter(path + '/run')

    real, label, _ = real_ds[1]
    # synth, label, _ = synth_ds[1]

    real_dl = DataLoader(real_ds, batch_size = batch_size, shuffle = True)
    synth_dl = DataLoader(synth_ds, batch_size = batch_size, shuffle = True)

    Gen_A2B = Generator().to(device)
    Gen_B2A = Generator().to(device)
    Dis_A2B = Discriminator().to(device)
    Dis_B2A = Discriminator().to(device)

    Gen_A2B.apply(initialize_weight)
    Gen_B2A.apply(initialize_weight)
    Dis_A2B.apply(initialize_weight)
    Dis_B2A.apply(initialize_weight)

    gan_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()

    opt_gen = optim.Adam(itertools.chain(Gen_A2B.parameters(), Gen_B2A.parameters()), lr = lr, betas = (0.5, 0.999))
    opt_dis = optim.Adam(itertools.chain(Dis_A2B.parameters(), Dis_B2A.parameters()), lr = lr, betas = (0.5, 0.999))
    sche_gen = optim.lr_scheduler.StepLR(opt_gen, step_size = 50, gamma = 0.5)
    sche_dis = optim.lr_scheduler.StepLR(opt_dis, step_size = 50, gamma = 0.5)

    Gen_A2B.train()
    Gen_B2A.train()
    Dis_A2B.train()
    Dis_B2A.train()

    fake_real_buffer = ReplayBuffer()
    fake_synth_buffer = ReplayBuffer()

    start_time = time.time()
    iter_num = int(num_epoch * len(real_dl))
    iter_count = 0
    print('Start Train')

    for i in range(iter_num):
        if not (i % len(real_dl)):
            real_set = iter(real_dl)
        
        if not (i % len(synth_dl)):
            synth_set = iter(synth_dl)

        real_img, _, _ = next(real_set)
        synth_img, _, _ = next(synth_set)

        real_img = real_img.to(device)
        synth_img = synth_img.to(device)
        # Discriminator #
        # A2B : Synth -> Real, B2A : Real -> Synth
        Dis_A2B.zero_grad()
        Dis_B2A.zero_grad()

        # Fake Loss
        fake_real = Gen_A2B(synth_img)
        fake_synth = Gen_B2A(real_img)
        
        fake_real_pop = fake_real_buffer.push_and_pop(fake_real)
        fake_synth_pop = fake_synth_buffer.push_and_pop(fake_synth)
        out_dis_A2B = Dis_A2B(fake_real_pop.detach())
        out_dis_B2A = Dis_B2A(fake_synth_pop.detach())
        fake_loss_A2B = gan_loss(out_dis_A2B, torch.zeros(out_dis_A2B.shape, requires_grad = False).to(device))
        fake_loss_B2A = gan_loss(out_dis_B2A, torch.zeros(out_dis_B2A.shape, requires_grad = False).to(device))

        # Real Loss
        out_dis_A2B = Dis_A2B(real_img)
        out_dis_B2A = Dis_B2A(synth_img)
        real_loss_A2B = gan_loss(out_dis_A2B, torch.ones(out_dis_A2B.shape, requires_grad = False).to(device))
        real_loss_B2A = gan_loss(out_dis_B2A, torch.ones(out_dis_B2A.shape, requires_grad = False).to(device))

        # Total Loss
        loss_A2B = (fake_loss_A2B + real_loss_A2B) / 2
        loss_B2A = (fake_loss_B2A + real_loss_B2A) / 2

        loss_A2B.backward()
        loss_B2A.backward()

        opt_dis.step()

        # Generator #
        Gen_A2B.zero_grad()
        Gen_B2A.zero_grad()

        # Gan Loss
        out_dis_A2B = Dis_A2B(fake_real)
        out_dis_B2A = Dis_B2A(fake_synth)
        gan_loss_A2B = gan_loss(out_dis_A2B, torch.ones(out_dis_A2B.shape, requires_grad = False).to(device))
        gan_loss_B2A = gan_loss(out_dis_B2A, torch.ones(out_dis_B2A.shape, requires_grad = False).to(device))

        # Cycle Consistency Loss
        recon_synth = Gen_B2A(fake_real)
        recon_real = Gen_A2B(fake_synth)

        cycle_synth = cycle_loss(recon_synth, synth_img) * 100.0
        cycle_real = cycle_loss(recon_real, real_img) * 100.0

        # Total Loss
        loss_G = gan_loss_A2B + gan_loss_B2A + cycle_synth + cycle_real

        loss_G.backward()

        opt_gen.step()

        # Tensorboard
        iter_count += 1
        writer.add_scalar("G_Loss/Synth2Real", gan_loss_A2B + cycle_real, iter_count)
        writer.add_scalar("G_Loss/Real2Synth", gan_loss_B2A + cycle_synth, iter_count)
        writer.add_scalar("D_Loss/Synth2Real", loss_A2B, iter_count)
        writer.add_scalar("D_Loss/Real2Synth", loss_B2A, iter_count)

        if not (i % len(real_dl)):
            epoch = int(i / len(real_dl))
            print('Epoch: {}, G_Loss: {:.2f}, D_Loss: {:.2f}, Time: {:3d} min'.format(epoch, loss_G, loss_A2B + loss_B2A, int((time.time() - start_time) / 60)))
            plot_fake_real = make_grid(fake_real * gamma + mu, nrow = 8, padding = 20, pad_value = 0.5)
            plot_synth = make_grid(synth_img * gamma + mu, nrow = 8, padding = 20, pad_value = 0.5)
            plot_real = make_grid(real_img * gamma + mu, nrow = 8, padding = 20, pad_value = 0.5)
            writer.add_image("Generatd_Image", plot_fake_real, iter_count)
            writer.add_image("Synthetic_Image", plot_synth, iter_count)
            writer.add_image("Real_Image", plot_real, iter_count)

            sche_gen.step()
            sche_dis.step()
        
    writer.close()

    # Save Weight(*.pt)
    os.makedirs(path, exist_ok = True)
    path2gen = os.path.join(path, 'weights_RBOX_rot_pad_CycleGAN.pt')
    torch.save(Gen_A2B.state_dict(), path2gen)
