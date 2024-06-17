import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import discriminator, generator, weights_init

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./checkpoints/CIFAR10', exist_ok=True)
os.makedirs('./runs', exist_ok=True)
os.makedirs('./runs/CIFAR10', exist_ok=True)
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)
'''
    创建两个数据增强
'''
image_size = 64
train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])


'''
创建数据迭代器
'''

train_file = datasets.CIFAR10(
    root='../分组作业4/data/',
    train=True,
    transform=train_transforms,
    download=True
)


batch_size = 64
nz, nc, ndf, ngf = 100, 3, 64, 64
epochs = 25
lr = 0.0002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
real_label = 1.
fake_label = 0.

netG = generator(nz, ngf, nc).to(device)
netD = discriminator(nc, ndf).to(device)
netD.apply(weights_init)
netG.apply(weights_init)
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

train_loader = DataLoader(train_file, batch_size=batch_size, shuffle=True)

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Start training...")
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        '''
            这里训练判别器很简单哈！就是让他平时多看看真的样本，然后再去看看generator生成出来的样本
        '''
        # Train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        # Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        '''
            这里训练generator，也比较容易理解，就是我希望我生成出来的图片不会被discriminater发现，我就是告诉discriminater我是真的，让他自己去猜
        '''
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.grid(False)
plt.legend()
plt.show()

np.savetxt('./checkpoints/CIFAR10/G_losses.txt', G_losses)
np.savetxt('./checkpoints/CIFAR10/D_losses.txt', D_losses)
torch.save(netG.state_dict(), './checkpoints/CIFAR10/netG.pt')
torch.save(netD.state_dict(), './checkpoints/CIFAR10/netD.pt')
writer = SummaryWriter('./runs/CIFAR10')
for step, img in enumerate(img_list):
    writer.add_image("image_grid", img, global_step=step*500)
writer.close()
