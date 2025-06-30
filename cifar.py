import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from cifar_models import cifar10_ddpmpp_continuous 
from cifar_models.ncsnpp import NCSNpp
import tqdm

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

class Learner:
    def __init__(self, device, data_loader, uncond_estimator_num = 4096,
                 alpha = lambda t: 1 - t, beta = lambda t: t, a = None, b = None,
                 denoising_ratio = 0.75, total_steps = 400000):
        
        self.device = device
        self.data_loader = data_loader
        self.uncound_estimator_num = uncond_estimator_num
        
        # self.data_list = []
        # self.count = 0
        # for images, labels in self.data_loader:
        #     self.count += images.size(0)
        #     self.data_list.append(images)
        #     if self.count >= self.uncound_estimator_num:
        #         self.count = 0
        #         break
            
        #self.uncound_estimator = torch.cat(self.data_list).to(self.device)       
    
        self.alpha = alpha
        self.beta = beta
        
        if a == None:
            self.a = lambda t, s: self.alpha(s) / (self.alpha(t) + 1e-4)
        else:
            self.a = a
            
        if b == None:
            self.b = lambda t, s: (self.alpha(t) * self.beta(s) - self.alpha(s) * self.beta(t)) / (self.alpha(t) + 1e-4) 
        else:
            self.b = b
        
        # Parameterization: f(x,t,s) = a(t,s)x + NN(x,t,s-t)
        config = cifar10_ddpmpp_continuous.get_config()  
        config.data.num_channels = 1
        self.network = NCSNpp(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=5e-4, betas=(0.9,0.999),weight_decay=0,eps=1e-8)
        
        self.denoising_ratio = denoising_ratio
        self.total_steps = total_steps
        self.now_step = 0 
    
    def alpha_grad(self, t: torch.Tensor):
        t = t.clone().detach()
        t.requires_grad = True
        alpha_value = self.alpha(t)
        alpha_value.sum().backward()
        return t.grad

    def beta_grad(self, t: torch.Tensor):
        t = t.clone().detach()
        t.requires_grad = True
        beta_value = self.beta(t)
        beta_value.sum().backward()
        return t.grad
    
    def a_grad(self, t: torch.Tensor, s: torch.Tensor):
        t = t.clone().detach()
        t.requires_grad = True
        s = s.clone().detach()
        s.requires_grad = True
        a_value = self.a(t, s)
        a_value.sum().backward()
        return t.grad, s.grad
        
    def b_grad(self, t: torch.Tensor, s: torch.Tensor):
        t = t.clone().detach()
        t.requires_grad = True
        s = s.clone().detach()
        s.requires_grad = True
        b_value = self.b(t, s)
        b_value.sum().backward()
        return t.grad, s.grad
    
    def train(self):
        while True:
            tqdm_loader = tqdm.tqdm(self.data_loader, desc=f'Start from {self.now_step} Steps')
            avg_stats = {'diff':None,'eq':None,'w':None,'grad':None}
            alpha = 0.98
            for images, labels in tqdm_loader:
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch = images.size(0)
                denoising_num = int(self.denoising_ratio * batch)
                equation_num = batch - denoising_num

                if denoising_num > 0:
                    images_denoising = images[:denoising_num]
                    labels_denoising = labels[:denoising_num]
                    x_1 = torch.randn_like(images_denoising)
                    t = torch.sigmoid(torch.randn(denoising_num, device = x_1.device) * 1.0 - 0.4)
                    t = t.view(denoising_num, 1, 1, 1)
                    x_t = self.alpha(t) * images_denoising + self.beta(t) * x_1
                    denoising_loss = (self.network(x_t,t,t) - x_1).pow(2).mean()                     
                else:
                    denoising_loss = 0
                     
                if equation_num > 0:
                    images_equation = images[denoising_num:]
                    labels_equation = labels[denoising_num:]
                    x_1 = torch.randn_like(images_equation)
                    time = torch.sigmoid(torch.randn(2 * equation_num, device = x_1.device) * 1.0 - 0.4)
                    t = torch.max(time[:equation_num], time[equation_num:])
                    s = torch.min(time[:equation_num], time[equation_num:])
                    t = t.view(equation_num, 1, 1, 1)
                    s = s.view(equation_num, 1, 1, 1)
                    x_t = self.alpha(t) * images_equation + self.beta(t) * x_1
                    v_t = self.alpha_grad(t) * images_equation + self.beta_grad(t) * x_1
                    F_value, jvp = torch.autograd.functional.jvp(
                        func = lambda x_t,t,s:self.network(x_t,t,s),
                        inputs = (x_t, t, s), 
                        v = (v_t, torch.ones_like(t), torch.zeros_like(s)),
                        create_graph = False, strict = True)
                    a, b = self.a(t, s), self.b(t, s)
                    papt, paps = self.a_grad(t, s)
                    pbpt, pbps = self.b_grad(t, s)
                    delta = (a * v_t + b * jvp + papt * x_t + pbpt * F_value).view(equation_num, -1)
                    w = 1.0 / (pbps * torch.norm(delta, p=2, dim=1).detach() + 1e-2)
                    equation_loss = (w * delta.pow(2).sum(dim=1)).mean()
                else:
                    equation_loss = 0
                
                loss = denoising_loss + equation_loss
                self.optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0, norm_type=2.0)
                self.optimizer.step()
                if avg_stats['grad'] == None:
                    avg_stats['diff'] = denoising_loss.item()
                    avg_stats['eq'] = equation_loss.item()
                    avg_stats['w'] = w.mean().item()
                    avg_stats['grad'] = grad.item()
                else:
                    avg_stats['diff'] = avg_stats['diff'] * alpha + (1 - alpha) * denoising_loss.item()
                    avg_stats['eq'] = avg_stats['eq'] * alpha + (1 - alpha) * equation_loss.item()
                    avg_stats['w'] = avg_stats['w'] * alpha + (1 - alpha) * w.mean().item()
                    avg_stats['grad'] = avg_stats['grad'] * alpha + (1 - alpha) * grad.item()
                    
                tqdm_loader.set_postfix(diff = avg_stats['diff'],
                                        eq = avg_stats['eq'],
                                        w = avg_stats['w'],
                                        grad = avg_stats['grad'])
                self.now_step += 1
                if self.now_step % 100 == 0:
                    with torch.inference_mode():
                        t = torch.ones((10,1,1,1), device = self.device)
                        s = torch.zeros((10,1,1,1), device = self.device)
                        x_t = torch.randn((10,1,32,32), device = self.device)
                        generated = self.solution_operator(x_t, t, s).clamp(min=-1,max=1)
                        os.makedirs('figs', exist_ok = True)
                        self.visualize_batch(f'figs/{self.now_step}.png',generated)
                        
                if self.now_step >= self.total_steps:
                    return 
                
    def get_loss(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, c: torch.LongTensor):
        # x: [B, C, H, W], t: [B, 1], s: [B, 1], c: [B, 1]
        torch.autograd.functional.jvp()
    
    def solution_operator(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        return self.a(t, s) * x_t + self.b(t, s) * self.network(x_t , t , s)
        
    def visualize_batch(self, save_name, images, labels=None, n_cols = 5):
        """ images: [batch_size, 3, 32, 32], labels:[batch_size]"""
        n = min(len(images), 10) 
        n_rows = (n + n_cols - 1) // n_cols
        plt.figure(figsize=(2 * n_cols, 2 * n_rows))
        #plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n):
            plt.subplot(n_rows, n_cols, i + 1)
            img = images[i].permute(1,2,0) / 2 + 0.5
            plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 1)
            if labels != None:
                plt.title(f"{classes[labels[i]]} ({labels[i]})")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_name)
          
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=4,
    shuffle=True, 
    num_workers=0
) 
# config = cifar10_ddpmpp_continuous.get_config()  
# config.data.num_channels = 1
# model = NCSNpp(config)
# def func(x,t,s):
#     return model.forward(x,t,s)
# input = direction = (torch.randn(3,1,32,32),torch.rand(3),torch.rand(3))
# direction = (torch.randn(3,1,32,32),torch.rand(3),torch.rand(3))
# predict, jvp = torch.autograd.functional.jvp(func,input,direction)
# print(predict.shape, jvp.shape)
# print(model().shape)
# print(config.data)
learner = Learner('cuda',trainloader)
learner.train()
# print(learner.alpha_grad(torch.randn(5)))
# print(learner.beta_grad(torch.randn(5)))
#print(learner.a_grad(torch.tensor(0.4),torch.tensor(0.2)))
#learner.train()
