import glob
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import trainer
import argparse
from utils import utils, hooks
from models.cifar_models import *
import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

def train_loop(model, device, train_loader, test_loader, optimizer, model_type='teacher', teacher_model=None, tensor_mask=None):
    test_acc_list=[]
    test_loss_list=[]
    mean_norm_list=[]
    layer_norm_list=[]
    scheduler=StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(40)):
        train_loss=trainer.student_train(model, device, train_loader, optimizer, teacher_model)
        test_loss, test_acc=trainer.test(model, device, test_loader)
        layer_norm, mean_norm=utils.model_norm(teacher_model, model, tensor_mask)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        layer_norm_list.append(layer_norm)
        mean_norm_list.append(mean_norm)
        print(f"Epoch {epoch} test acc {test_acc}")
        scheduler.step()

    return test_acc_list, test_loss_list, layer_norm_list, mean_norm_list


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

    #torch.manual_seed(args.seed)

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
    #model_list=[Base()]

    for model_index, model in enumerate(model_list):
        model_save_path=os.path.join('saved_models', 'cifar', model.name)

        model=torch.load(model_save_path)
        
        teacher_test_loss, teacher_test_acc=trainer.test(model, device, test_loader)
        print(f'Teacher Test Loss {teacher_test_loss}; Test Acc {teacher_test_acc}')
        state_dict=model.state_dict()
        tensor_mask=[]

        new_state_dict=OrderedDict()
        for layer, weight in state_dict.items():
            new_state_dict[layer], mask=utils.tensor_corruptor(weight, all_weights=True)
            count=torch.numel(mask[mask==1])
            print(f"layer: {layer}; percentage: {count/torch.numel(mask)}")
            tensor_mask.append(mask.to(device))

        student_model = model_list[model_index].to(device)
        student_model.load_state_dict(new_state_dict)
        student_model = student_model.to(device)

        print("Norm after adhoc error correction")
        utils.model_norm(model, student_model, tensor_mask)

        for i, (name, param) in enumerate(student_model.named_parameters()):
            param.register_hook(hooks.random_dropout)
            #param.register_hook(lambda grad, i=i: grad*tensor_mask[i])
        #student_test_loss, student_test_acc=trainer.test(student_model, device, test_loader)
        #student_test_acc_list.append(test_acc-student_test_acc)

        #optimizer = optim.Adam(student_model.parameters(), lr=1e-6)

        student_train_loader = DataLoader(
            datasets.STL10('../data', split='unlabeled', download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=64, **kwargs)

        student_data= datasets.STL10('../data', split='unlabeled', download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ]))

        split=True
        if split:
            p=0.1
            train_end=np.arange(0, int(p*len(student_data)))
            test_end=np.arange(int(p*len(student_data)), int((p+0.02)*len(student_data)))
            student_train_data_subset=torch.utils.data.Subset(student_data, train_end)
            student_test_data_subset=torch.utils.data.Subset(student_data, test_end)

            student_train_loader=DataLoader(student_train_data_subset, batch_size=128)
            student_test_loader=DataLoader(student_test_data_subset, batch_size=32)
        else:
            student_train_loader=DataLoader(student_train_data, batch_size=32, shuffle=True)

        print("Training Student Model on Random Data")

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=1e-4)
        test_acc_list, test_loss_list, layer_norm_list, mean_norm_list = train_loop(student_model, device,\
                    student_train_loader, test_loader, optimizer, model_type='student', teacher_model=model, tensor_mask=tensor_mask)

        rad_test_acc_list=[(teacher_test_acc-acc)/teacher_test_acc for acc in test_acc_list]
        if not os.path.exists(os.path.join('recovery_data', 'cifar', 'no_data_10', 'drop', model.name.split('.')[0])):
            os.makedirs(os.path.join('recovery_data', 'cifar','no_data_10',  'drop',  model.name.split('.')[0]))
        np.save(os.path.join('recovery_data', 'cifar','no_data_10',   'drop', model.name.split('.')[0], 'rad_test_acc_list'), rad_test_acc_list)
        np.save(os.path.join('recovery_data', 'cifar', 'no_data_10', 'drop',  model.name.split('.')[0], 'mean_norm'), mean_norm_list)
        np.save(os.path.join('recovery_data', 'cifar', 'no_data_10', 'drop',  model.name.split('.')[0], 'layer_norm_list'), layer_norm_list)

if __name__ == '__main__':
    main()
