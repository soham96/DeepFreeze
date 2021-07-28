import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def randomizer(number):
    mask=0
    num=float_to_bin(number)
    random_list=np.random.random_sample(size=32)
    #one_to_zero=random_list<=0.001
    #zero_to_one=random_list<=0.0001
    one_to_zero=random_list<=0.0000027
    zero_to_one=random_list<=0.00000009
    num=['0' if otz else n for n, otz in zip(num, one_to_zero)]
    num=['1' if zto else n for n, zto in zip(num, zero_to_one)]
    
    num=''.join(num)
    if num[1:9]=='11111111':
        num=num[0]+'0'+num[2:]

    new_number=bin_to_float(num)
    #if int(new_number)>100:
    #    print(f"{number} --> {new_number}")
    #import ipdb; ipdb.set_trace()
    if int(abs(new_number))>10:
        #import ipdb; ipdb.set_trace()
        #new_number=new_number/10**(len(str(int(new_number)))+1)
        new_number=0
        mask=1
    #if abs(new_number)>10:
    #    new_number=0
    #    mask=1
    if abs(new_number)<1e-4:
        #import ipdb; ipdb.set_trace()
        #new_number=new_number*10**(len(str(int(new_number)))+2)
        new_number=0
        mask=1
    return new_number, mask

def tensor_corruptor(tensor, all_weights=False):
    original_shape=tensor.shape
    new_tensor=[]
    tensor_mask=[]
    tensor=tensor.flatten()

    for t in tensor:
        corrupted_tensor, mask=randomizer(t)
        new_tensor.append(corrupted_tensor)
        tensor_mask.append(mask)

    new_tensor=torch.tensor(new_tensor, dtype=torch.float32)
    tensor_mask=torch.tensor(tensor_mask)
    return new_tensor.reshape(original_shape),  \
            tensor_mask.reshape(original_shape)

def layer_corruptor(tensor, k=0.01):
    original_shape=tensor.shape
    mask=np.random.random_sample(size=original_shape)
    mask[mask<=k]=0
    mask[mask>k]=1

    mask=mask.astype(np.float32)
    new_tensor=tensor.cpu()*mask

    mask=1-mask
    new_tensor=torch.Tensor(new_tensor)
    tensor_mask=torch.Tensor(mask)
    return new_tensor, tensor_mask

def model_norm(teacher_model, student_model, tensor_mask):
    teacher_model.eval()
    student_model.eval()

    tm_state_dict=teacher_model.state_dict()
    sm_state_dict=student_model.state_dict()
    layer_norm_list=[]
    non_mask_layer_norm_list=[]
    layer_mae_list=[]
    non_mask_layer_mae_list=[]
    layer_count=len(tm_state_dict)
    count=0

    for (tm_layer, tm_weight), (sm_layer, sm_weight), mask in zip(tm_state_dict.items(), sm_state_dict.items(), tensor_mask):
        if 0 in mask:
            #mask=mask.to(tm_weight.device)
            layer_norm=torch.norm((tm_weight-sm_weight)*mask)/torch.norm(tm_weight*mask)
            non_masked_norm=torch.norm((tm_weight-sm_weight))/torch.norm(tm_weight)
            mae=torch.sum(abs(tm_weight-sm_weight))/torch.numel(tm_weight)
            masked_mae=torch.sum(abs(tm_weight-sm_weight)*mask)/torch.numel(tm_weight)
            #layer_norm=torch.norm(torch.div((tm_weight-sm_weight),tm_weight)*mask)
            layer_norm_list.append(layer_norm)
            non_mask_layer_norm_list.append(non_masked_norm)
            non_mask_layer_mae_list.append(mae)
            layer_mae_list.append(masked_mae)
            #print(f'For layer {sm_layer}; Masked Norm {layer_norm}; Unmasked Norm {non_masked_norm}')
            #print(f'For layer {sm_layer}; Masked MAE {masked_mae}; Unmasked MAE {mae}')
            #print(f"Num: {torch.norm((tm_weight-sm_weight)*mask)}, Denom: {torch.norm(tm_weight*mask)}")

    #print(f"Mean Norm for all layers {sum(layer_norm_list)/len(layer_norm_list)}")
    #print(f"Mean Non-Masked Norm for all layers {sum(non_mask_layer_norm_list)/len(non_mask_layer_norm_list)}")
    #print(f"Mean Masked MAE for all layers {sum(layer_mae_list)/len(layer_mae_list)}")
    #print(f"Mean Non-Masked MAE for all layers {sum(non_mask_layer_mae_list)/len(non_mask_layer_mae_list)}")
    #print(f"Total Layers: {layer_count}; Norm Layers: {len(layer_norm_list)}")

    non_mask_layer_norm_list=[i.item() for i in non_mask_layer_norm_list]
    return non_mask_layer_norm_list, sum(non_mask_layer_norm_list)/len(non_mask_layer_norm_list)

def dataloader(data):

    if data=='mnist':
        train_loader = DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([ transforms.ToTensor(),
                    #transforms.Normalize((0.1307), (0.3081))
                    ])),
                    batch_size=64)
        test_loader = DataLoader(
            datasets.MNIST('../data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.1307), (0.3081))
                           ])),
                batch_size=64)
    elif data=='cifar':
        train_loader = DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                           transform=transforms.Compose([
                               #transforms.RandomCrop(32, padding=4),
                               transforms.Resize((224, 224), interpolation=Image.NEAREST),
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
                               transforms.Resize((224, 224), interpolation=Image.NEAREST),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=32)

    return train_loader, test_loader
