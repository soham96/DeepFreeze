from __future__ import print_function
from collections import OrderedDict
from tqdm import tqdm
import os
import numpy as np
import trainer
import argparse
from models.cifar_models import *
from utils import utils, hooks
import torch
from torch.utils.data import DataLoader 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

def train_loop(model, device, train_loader, test_loader, optimizer, model_type='teacher', teacher_model=None):
    count=0
    prev_test_loss=0
    train_acc=0
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in tqdm(range(50)):
        if model_type=='student':
            train_loss=trainer.student_train(model, device, train_loader, optimizer, teacher_model)
            test_loss, test_acc=trainer.test(model, device, test_loader)
            print(f'Epoch {epoch} Training Loss {train_loss}; Test Loss {test_loss}; Test Acc {test_acc}')
            break
        else:
            train_loss, train_acc=trainer.train(model, device, train_loader, optimizer)
            test_loss, test_acc=trainer.test(model, device, test_loader)
            scheduler.step()
            print(f'Epoch {epoch} Training Accuracy: {train_acc}; Training Loss {train_loss}')
            print(f'Epoch {epoch} Testing Accuracy: {test_acc}; Testing Loss {test_loss}')
        if prev_test_loss==0:
            prev_test_loss=test_loss
        if test_loss>prev_test_loss:
            count+=1
        prev_test_loss=test_loss
        if count>2:
            break

    return train_loss, train_acc, test_loss, test_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on {device}")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs={}
    train_loader = DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(p=0.5),
                           #transforms.RandomRotation(10),
                           #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), 
	                   #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=32)
    test_loader = DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=32)

    model_list=[Base(), BaseWide(), BaseDropout(), BasePReLU(), LeNet5(), LeNet5Dropout()]

    for model_index, model in enumerate(model_list):
        model_save_path=os.path.join('saved_models', 'cifar', model.name)
        print(model_save_path)
        try:
            model=torch.load(model_save_path)
            print('loaded model')
        except:
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.1)

            train_loss, train_acc, test_loss, test_acc=train_loop(model, device,\
                    train_loader, test_loader, optimizer)
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            torch.save(model, model_save_path)
        
        teacher_test_loss, teacher_test_acc=trainer.test(model, device, test_loader)
        print(f'Teacher Test Loss {teacher_test_loss}; Test Acc {teacher_test_acc}')

        rad_list=[]
        mean_norm_list=[]

        for _ in tqdm(range(20)):
            state_dict=model.state_dict()
            tensor_mask=[]
            new_state_dict=OrderedDict()
            for layer, weight in state_dict.items():
                new_state_dict[layer], mask=utils.tensor_corruptor(weight)
                tensor_mask.append(mask.to(device))

            student_model = model_list[model_index].to(device)
            student_model.load_state_dict(new_state_dict)
            student_test_loss, student_test_acc=trainer.test(student_model, device, test_loader)

            rad=(teacher_test_acc-student_test_acc)/teacher_test_acc
            _, mean_norm=utils.model_norm(model, student_model, tensor_mask)

            rad_list.append(rad)
            mean_norm_list.append(mean_norm)
        
        if not os.path.exists(os.path.join('model_data', 'cifar', model.name.split('.')[0])):
            os.makedirs(os.path.join('model_data', 'cifar', model.name.split('.')[0]))
        np.save(os.path.join('model_data', 'cifar', model.name.split('.')[0], 'rad'), rad_list)
        np.save(os.path.join('model_data','cifar',  model.name.split('.')[0], 'mean_norm'), mean_norm_list)

if __name__ == '__main__':
    main()
