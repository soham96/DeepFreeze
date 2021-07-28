from __future__ import print_function
from collections import OrderedDict
import numpy as np
import trainer
import argparse
from models import Base
from utils import utils, hooks
import torch
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

class ModelDataGen(IterableDataset):
    def __init__(self, model, loader=None, shape=None):
        self.model=model
        self.input_shape=shape
        self.count=0
        self.loader=loader

    def __iter__(self):
        return self

    def __len__(self):
        return self.count

    def __next__(self):
        self.count+=1
        if self.count>100:
            raise StopIteration
        #random_input=np.random.standard_normal(size=self.input_shape)
        random_input=np.random.uniform(size=self.input_shape)
        model_input=np.expand_dims(random_input, axis=0)
        model_input=torch.Tensor(model_input)
        output=self.model(model_input)
        return torch.from_numpy(random_input).float(), F.log_softmax(output, dim=1)

def train_loop(model, device, train_loader, test_loader, optimizer, model_type='teacher', teacher_model=None, tensor_mask=None):
    count=0
    prev_test_loss=0
    train_acc=0
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(50):
        if model_type=='student':
            train_loss=trainer.student_train(model, device, train_loader, optimizer, teacher_model)
            test_loss, test_acc=trainer.test(model, device, test_loader)
            print(f'Epoch {epoch} Training Loss {train_loss}; Test Loss {test_loss}; Test Acc {test_acc}')
            utils.model_norm(teacher_model, model, tensor_mask)
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
        #if count>2:
        #    break

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
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.RandomRotation(10),
                           transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), 
	                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transforms.RandomCrop(32, padding=4),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size)
    test_loader = DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size)

    model_save_path='teacher_model.pt'
    try:
        model=torch.load(model_save_path)
    except:
        model = Base().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)

        train_loss, train_acc, test_loss, test_acc=train_loop(model, device,\
                train_loader, test_loader, optimizer)
        torch.save(model, model_save_path)
    
    test_loss, test_acc=trainer.test(model, device, test_loader)
    print(f'Teacher Test Loss {test_loss}; Test Acc {test_acc}')
    state_dict=model.state_dict()
    student_test_acc_list=[]
    tensor_mask=[]

    #for _ in tqdm(range(2)):
    new_state_dict=OrderedDict()
    layer_num=10
    for i, (layer, weight) in enumerate(state_dict.items()):
        #if layer_num==i:
        print(layer)
        new_state_dict[layer], mask=utils.tensor_corruptor(weight, all_weights=True)
        #_ , mask=utils.tensor_corruptor(weight)
        count=torch.numel(mask[mask==1])
        print(f"layer: {layer}; percentage: {count/torch.numel(mask)}")
        #new_state_dict[layer], mask=utils.layer_corruptor(weight, k=count/torch.numel(mask))
        #mask=mask.to(device)
        tensor_mask.append(mask.to(device))
        #else:
        #    weight=weight.to(device)
        #    new_state_dict[layer]=weight
        #    mask=torch.zeros(weight.shape)
        #    mask=mask.to(device)
        #    tensor_mask.append(mask)

    student_model = Base().to(device)
    student_model.load_state_dict(new_state_dict)

    print("Norm after adhoc error correction")
    utils.model_norm(model, student_model, tensor_mask)

    #for i, (name, param) in enumerate(student_model.named_parameters()):
        #param.register_hook(hooks.myhook)
        #param.register_hook(lambda grad, i=i: grad*tensor_mask[i])

    student_test_loss, student_test_acc=trainer.test(student_model, device, test_loader)
    student_test_acc_list.append(test_acc-student_test_acc)

    acc_drop=np.asarray(student_test_acc_list).mean()
    print(f"Testing Acc Drop: {acc_drop}")
    np.save('CIFAR10_1000corruptions.npy', np.asarray(student_test_acc_list))


    #optimizer = optim.Adam(student_model.parameters(), lr=1e-6)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    student_train_loader = DataLoader(
        datasets.STL10('../data', split='train+unlabeled', download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=32, **kwargs)

    #student_train_loader=DataLoader(ModelDataGen(model, (3, 32, 32)), batch_size=args.batch_size)

    print("Training Student Model on Random Data")
    for (name, param), mask in zip(student_model.named_parameters(), tensor_mask):
        if 1 not in mask:
            continue
        for p in student_model.parameters():
            # Change to False to enable layerwise training
            p.requires_grad = False
        param.requires_grad=True
        for i in range(200):
            print(f"Epoch {i}")
            student_train_loss, student_train_acc, student_test_loss, student_test_acc=train_loop(student_model, device,\
                    student_train_loader, test_loader, optimizer, model_type='student', teacher_model=model, tensor_mask=tensor_mask)

            print("Norm after KD training")
            utils.model_norm(model, student_model, tensor_mask)

    student_train_loss, student_train_acc, student_test_loss, student_test_acc=train_loop(student_model, device,\
            student_train_loader, test_loader, optimizer, model_type='student', teacher_model=model)

    print("Norm after KD training")
    utils.model_norm(model, student_model, tensor_mask)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
