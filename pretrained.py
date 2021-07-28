from collections import OrderedDict
from tqdm import tqdm
import os
import numpy as np
import trainer
import argparse
from models.pretrained_models import pretrained_model
from utils import utils, hooks
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

def train_loop(model, device, dataloader, dataloader_size, optimizer, model_type='teacher', teacher_model=None):
    count=0
    prev_test_loss=0
    train_acc=0
    for epoch in tqdm(range(20)):
        if model_type=='student':
            train_loss=trainer.student_train(model, device, dataloader['train'], optimizer, teacher_model)
            test_loss, test_acc=trainer.test(model, device, dataloader['val'])
            print(f'Epoch {epoch} Training Loss {train_loss}; Test Loss {test_loss}; Test Acc {test_acc}')
            break
        else:
            #import ipdb; ipdb.set_trace()
            train_loss, train_acc=trainer.train(model, device, dataloader['train'], optimizer)
            test_loss, test_acc=trainer.test(model, device, dataloader['test'])
            #print(f'Epoch {epoch} Training Accuracy: {train_acc}; Training Loss {train_loss}')
            print(f'Epoch {epoch} Testing Accuracy: {test_acc}; Testing Loss {test_loss}')

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

    #data_dir = 'pytorch-tiny-imagenet/tiny-224/'
    data_dir = '../data/flower_data/'
    num_workers = {'train' : 100,'val'   : 0,'test'  : 0}
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
                      for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    model_list=['alexnet', 'vgg', 'resnet']

    for model_index, model_name in enumerate(model_list):
        model_save_path=os.path.join('saved_models', 'pretrained', model_name +'.pt')
        print(model_save_path)
        try:
            model=torch.load(model_save_path)
            print('loaded model')
        except:
            model, input_size = pretrained_model(model_name)
            model = model.to(device)
            print([p.requires_grad for p in model.parameters()])
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

            train_loss, train_acc, test_loss, test_acc=train_loop(model, device, dataloaders, dataset_sizes, optimizer)
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            torch.save(model, model_save_path)
        
        teacher_test_loss, teacher_test_acc=trainer.test(model, device, dataloaders['val'])
        print(f'Teacher Test Loss {teacher_test_loss}; Test Acc {teacher_test_acc}')

        rad_list=[]
        mean_norm_list=[]

        #for _ in tqdm(range(100)):
        #    state_dict=model.state_dict()
        #    tensor_mask=[]
        #    new_state_dict=OrderedDict()
        #    for layer, weight in state_dict.items():
        #        new_state_dict[layer], mask=utils.tensor_corruptor(weight)
        #        _ , mask=utils.tensor_corruptor(weight)
        #        count=torch.numel(mask[mask==1])
        #        #print(f"layer: {layer}; percentage: {count/torch.numel(mask)}")
        #        new_state_dict[layer], mask=utils.layer_corruptor(weight, k=count/torch.numel(mask))
        #        tensor_mask.append(mask.to(device))

        #    student_model = model_list[model_index].to(device)
        #    student_model.load_state_dict(new_state_dict)
        #    student_test_loss, student_test_acc=trainer.test(student_model, device, dataloaders['val'])

        #    rad=(teacher_test_acc-student_test_acc)/teacher_test_acc
        #    _, mean_norm=utils.model_norm(model, student_model, tensor_mask)

        #    rad_list.append(rad)
        #    mean_norm_list.append(mean_norm)
        #
        #if not os.path.exists(os.path.join('model_data', 'mnist', model.name.split('.')[0])):
        #    os.makedirs(os.path.join('model_data', 'mnist', model.name.split('.')[0]))
        #np.save(os.path.join('model_data', 'mnist', model.name.split('.')[0], 'rad'), rad_list)
        #np.save(os.path.join('model_data', 'mnist', model.name.split('.')[0], 'mean_norm'), mean_norm_list)


if __name__ == '__main__':
    main()
