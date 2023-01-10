import torch
import torchvision.utils
from torchvision import  transforms, datasets
from torch.utils.data import  Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import  SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from model import Generator, Discriminator
from PIL import Image
#Hyperparameters
lr_ = 0.001
noize_dim = 128
img_dim = 1*28*28
batch_size = 64
epochs = 100
sumary_real = SummaryWriter('runs/MNIST_real')
sumary_fake = SummaryWriter('runs/MNIST_fake')
step=0
##################################
fix_noise = torch.randn(40,128)
generator = Generator(noize_dim,img_dim)
discriminator = Discriminator(img_dim)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_)
optimizar_generator = optim.Adam(generator.parameters(), lr=lr_)

transforms_ = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataset_ = datasets.MNIST(root='MNIS_GAN', transform=transforms_,download=True)
data_loader = DataLoader(dataset=dataset_, shuffle=True, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
for epoch in range(epochs):
    for _, (imge,lable) in enumerate(data_loader):
        noise_input = torch.randn(batch_size, noize_dim)
        fake=generator(noise_input)
        real=discriminator(imge.reshape(-1, 28*28))
        output=discriminator(fake.detach())
        loss_dis=criterion(output, torch.zeros_like(output))+criterion(real, torch.ones_like(real))
        optimizer_discriminator.zero_grad()
        loss_dis.backward()
        optimizer_discriminator.step()

        #########################
        output = discriminator(fake)

        loss_gen = criterion(output, torch.ones_like(output))
        optimizar_generator.zero_grad()
        loss_gen.backward()
        optimizar_generator.step()
        if _ % 100 == 0:
            print(f'in the {epoch},the dis loss is {loss_dis:4f}, the gen loss is {loss_gen:4f}')
        with torch.no_grad():
            fake_image=generator(fix_noise).reshape(-1,1,28,28)
            real_img = imge.reshape(-1,1,28,28)
            fake_image_grid = torchvision.utils.make_grid(fake_image,normalize=True)
            real_img_grid = torchvision.utils.make_grid(real_img,normalize=True)

            # # print(type(fake_image_grid))
            # # plt.savefig(fake_image_grid.permute(1, 2, 0))
            # sumary_real.add_image('MNIST real images', real_img_grid,global_step=step)
            # sumary_fake.add_image('MNIS fake images', fake_image_grid,global_step=step)
            # step+=1
    img ='gen/img'+str(epoch)+'.jpg'
    save_image(fake_image_grid, img)

