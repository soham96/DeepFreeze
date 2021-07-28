import numpy as np
import os
import torch

def myhook(grad):
    grad_clone=grad.clone()

    grad_clone_shape=grad_clone.shape
    #import ipdb; ipdb.set_trace()

    #if grad_clone_shape==torch.Size([10]):
    #    return grad_clone

    grad_clone=grad_clone.flatten()
    bottom_k=int(0.10*len(grad_clone))
    #grad_clone[np.argpartition(abs(grad_clone), bottom_k)[:bottom_k]]=0
    grad_clone[torch.topk(abs(grad_clone), bottom_k, largest=False)[1]]=0

    #if os.path.exists('grad.npy'):
    #    saved_grad=np.load('grad.npy')
    #    saved_grad+=grad_clone.cpu().detach().numpy()
    #    np.save('grad.npy', saved_grad)
    #else:
    #    np.save('grad.npy', grad_clone.cpu().detach().numpy())
        

    grad_clone=grad_clone.reshape(grad_clone_shape)
    #np.save('grad.npy', grad_clone.cpu().detach().numpy())

    return grad_clone

def random_dropout(grad):
    grad_clone=grad.clone()

    grad_clone_shape=grad_clone.shape

    #if grad_clone_shape==torch.Size([10]):
    #    return grad_clone

    grad_clone=grad_clone.flatten()
    p=int(0.50*len(grad_clone))
    indices=np.random.choice(np.arange(len(grad_clone)), replace=False, size=p)
    grad_clone[indices]=0
    #grad_clone[np.argpartition(abs(grad_clone), bottom_k)[:bottom_k]]=0
    #grad_clone[torch.topk(abs(grad_clone), bottom_k, largest=False)[1]]=0

    #if os.path.exists('grad.npy'):
    #    saved_grad=np.load('grad.npy')
    #    saved_grad+=grad_clone.cpu().detach().numpy()
    #    np.save('grad.npy', saved_grad)
    #else:
    #    np.save('grad.npy', grad_clone.cpu().detach().numpy())
        

    grad_clone=grad_clone.reshape(grad_clone_shape)
    #np.save('grad.npy', grad_clone.cpu().detach().numpy())

    return grad_clone
def myhook(grad):
    grad_clone=grad.clone()

    grad_clone_shape=grad_clone.shape
    #import ipdb; ipdb.set_trace()

    #if grad_clone_shape==torch.Size([10]):
    #    return grad_clone

    grad_clone=grad_clone.flatten()
    bottom_k=int(0.10*len(grad_clone))
    #grad_clone[np.argpartition(abs(grad_clone), bottom_k)[:bottom_k]]=0
    grad_clone[torch.topk(abs(grad_clone), bottom_k, largest=False)[1]]=0

    #if os.path.exists('grad.npy'):
    #    saved_grad=np.load('grad.npy')
    #    saved_grad+=grad_clone.cpu().detach().numpy()
    #    np.save('grad.npy', saved_grad)
    #else:
    #    np.save('grad.npy', grad_clone.cpu().detach().numpy())
        

    grad_clone=grad_clone.reshape(grad_clone_shape)
    #np.save('grad.npy', grad_clone.cpu().detach().numpy())

    return grad_clone
