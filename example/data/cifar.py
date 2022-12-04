import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from utility.cutout import Cutout

def noisy_transform_img_help(prob=0.4):
    def noisy_transform_img(img):
        p1 = random.uniform(0, 1)
        if p1 < prob:
            p2 = random.uniform(0, 1)
            if p2 < 1/3:
                return transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            elif p2 < 2/3:
                return transforms.GaussianBlur(kernel_size=15, sigma=(1., 2.))(img)
            else:
                # mask =  torch.rand(image.shape, device=torch.device("cuda")) < frequency
                mask =  torch.rand(img.shape) < 0.05
                # new_vals = (torch.rand(image.shape, device=torch.device("cuda")) < 0.5).float()
                new_vals = (torch.rand(img.shape) < 0.5).float()
                img[mask] = new_vals[mask]
                return img
        else:
            return img
    return noisy_transform_img

class Cifar:

    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            noisy_transform_img_help(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
