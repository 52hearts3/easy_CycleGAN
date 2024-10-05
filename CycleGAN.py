import torch
from torch import nn,optim,autograd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.block=nn.Sequential(
            nn.ReflectionPad2d(1), #nn.ReflectionPad2d(1) 会在输入图像的每一边添加1个像素的反射填充,这样可以减少边缘效应，保持图像的边缘信息
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=0),
            nn.InstanceNorm2d(ch_out),
            #与批量归一化（Batch Normalization）不同，实例归一化（Instance Normalization）是在每个样本的每个通道上独立进行归一化操作，
            # 而不是在整个批量上进行。这在风格迁移和生成对抗网络（GAN）等任务中非常有用
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=0),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(True)
        )

        self.shortcut=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=0),
            nn.InstanceNorm2d(ch_out)
        )

    def forward(self,x):
        out=self.block(x)
        output=out+self.shortcut(x)
        return output

#test
test=torch.randn(12,3,64,64)
model=ResBlk(ch_in=3,ch_out=6)
print(model(test).shape)

class Generator(nn.Module):
    def __init__(self,ch_in=3,ch_out=3):
        super(Generator,self).__init__()

        out_channels=16
        self.conv1=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch_in,out_channels,kernel_size=7,stride=1,padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.down_sample=nn.Sequential(
            nn.Conv2d(out_channels , out_channels * 8,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(out_channels * 8),
            nn.ReLU(),
            nn.Conv2d(out_channels * 8, out_channels * 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 16),
            nn.ReLU(True),
        )

        self.res_block=nn.Sequential(
            ResBlk(out_channels * 16 , out_channels * 16),
            ResBlk(out_channels * 16 , out_channels * 16),
            ResBlk(out_channels * 16 , out_channels * 16),
            ResBlk(out_channels * 16 , out_channels * 16)
        )

        self.up_sample=nn.Sequential(
            nn.ConvTranspose2d(out_channels * 16,out_channels * 8,kernel_size=3,stride=2,padding=1,output_padding=1), #[b,ch,x,x]==>[b,ch,2x,2x]
            nn.InstanceNorm2d(out_channels*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels * 8, out_channels * 4, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.InstanceNorm2d(out_channels * 4),
            nn.ReLU(True)
        )

        # 计算量更大的版本，但是可以增加分辨率
        # self.up_sample = nn.Sequential(
        #    #  PixelShuffle(2) 是一种上采样方法，它通过重新排列通道来增加空间分辨率，同时减少通道数。
        #     nn.Conv2d(out_channels * 16, out_channels * 64, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2), #[b,ch,x,x]==>[b,ch/(2^2),2x,2x]  nn.PixelShuffle(x)--ch==>ch/(x^2)
        #     nn.InstanceNorm2d(out_channels * 16),
        #     nn.ReLU(True),
        #     nn.Conv2d(out_channels * 16, out_channels * 16, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.InstanceNorm2d(out_channels * 4),
        #     nn.ReLU(True)
        # )

        self.output_layer=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_channels * 4,ch_out,kernel_size=7,stride=1,padding=0),
            nn.Tanh()
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.down_sample(x)
        x=self.res_block(x)
        x=self.up_sample(x)
        x=self.output_layer(x)
        return x

test=torch.randn(32,3,64,64)
model=Generator(3,3)
print(model(test).size())

class Discriminator(nn.Module):
    def __init__(self,ch_in=3):
        super(Discriminator,self).__init__()

        self.block=nn.Sequential(
            nn.Conv2d(ch_in,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.output_layer=nn.Sequential(
            nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1),
            nn.AdaptiveAvgPool2d((1,1))  #[b,1,x,x]==>[b,1,1,1]
        )

    def forward(self,x):
        x=self.block(x)
        x=self.output_layer(x)
        x=x.view(x.size(0),-1)
        return x

#test
test=torch.randn(32,3,64,64)
model=Discriminator(3)
print(model(test).size())
import matplotlib.pyplot as plt

def show_last_image(image_tensor):
    # 将图像从Tensor转换为NumPy数组
    image = image_tensor.cpu().detach().numpy()
    # 调整维度顺序，从 (C, H, W) 到 (H, W, C)
    image = image.transpose(1, 2, 0)
    # 反归一化图像（在预处理时进行了归一化）
    image = (image * 0.5) + 0.5
    # 显示图像
    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()


tf=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_A=datasets.ImageFolder(root=r'D:\game\pytorch\GAN\GAN实战-GD实现\CycleGAN\horse2zebra\trainA',transform=tf)
dataset_B=datasets.ImageFolder(root=r'D:\game\pytorch\GAN\GAN实战-GD实现\CycleGAN\horse2zebra\trainB',transform=tf)
dataloader_A=DataLoader(dataset_A,batch_size=1,shuffle=True)
dataloader_B=DataLoader(dataset_B,batch_size=1,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_G_AtoB=Generator().to(device)
net_G_BtoA=Generator().to(device)
net_D_A=Discriminator().to(device)
net_D_B=Discriminator().to(device)

# 定义损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

import itertools
#itertools.chain，我们可以将两个生成器的参数连接起来，并使用一个优化器同时更新它们
optimizerG = optim.Adam(itertools.chain(net_G_AtoB.parameters(), net_G_BtoA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizerD_A = optim.Adam(net_D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD_B = optim.Adam(net_D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
lambda_cycle = 10.0
lambda_identity = 5.0
for epoch in range(1000):
    for data_A,data_B in zip(dataloader_A,dataloader_B):
        real_A,_ = data_A
        real_B,_ = data_B
        real_A,real_B=real_A.to(device),real_B.to(device)

        fake_B = net_G_AtoB(real_A)
        fake_A = net_G_BtoA(real_B)

        #计算生成器损失  训练生成器
        loss_GAN_AtoB=criterion_GAN( net_D_B(fake_B) , torch.ones_like(net_D_B(fake_B) ))
        loss_GAN_BtoA=criterion_GAN( net_D_A(fake_A) , torch.ones_like(net_D_A(fake_A) ))

        #计算循环一致性损失
        recover_A = net_G_BtoA(fake_B) #将生成的B图像转回A
        recover_B = net_G_AtoB(fake_A) #将生成的图像A转回B
        loss_cycle_A=criterion_cycle(recover_A,real_A)
        loss_cycle_B=criterion_cycle(recover_B,real_B)

        #计算身份损失
        loss_identity_A=criterion_identity( net_G_BtoA(real_A) , real_A) #我们希望 net_G_BtoA 在输入已经是域A的图像时，不改变图像内容
        loss_identity_B=criterion_identity( net_G_AtoB(real_B) , real_B) #我们希望 net_G_AtoB 在输入已经是域B的图像时，不改变图像内容

        #总生成器损失
        loss_G= loss_GAN_AtoB + loss_GAN_BtoA + lambda_cycle*(loss_cycle_A+loss_cycle_B) + lambda_identity*(loss_identity_A+loss_identity_B)

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        #训练判别器
        #计算判别器A的损失
        output_real_A=net_D_A(real_A)
        output_fake_A=net_D_A(fake_A.detach())
        loss_D_real_A = criterion_GAN( output_real_A , torch.ones_like(output_real_A) )
        loss_D_fake_A = criterion_GAN( output_fake_A , torch.zeros_like(output_fake_A) )
        loss_D_A=(loss_D_real_A+loss_D_fake_A)*0.5

        #计算判别器B的损失
        output_real_B=net_D_B(real_B)
        output_fake_B=net_D_B(fake_B.detach())
        loss_D_real_B = criterion_GAN( output_real_B , torch.ones_like(output_real_B) )
        loss_D_fake_B = criterion_GAN( output_fake_B , torch.zeros_like(output_fake_B) )
        loss_D_B=(loss_D_real_B+loss_D_fake_B)*0.5

        # 总判别器损失
        loss_D = loss_D_A + loss_D_B

        # 更新判别器
        optimizerD_A.zero_grad()
        loss_D_A.backward()
        optimizerD_A.step()

        optimizerD_B.zero_grad()
        loss_D_B.backward()
        optimizerD_B.step()

    print(f'[{epoch}/{1000}] Loss_D: {loss_D_A + loss_D_B} Loss_G: {loss_G}')
    show_last_image(fake_A[0])