import torch
import numpy as np
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data=data.reshape(-1, 28, 28, 1).to(device)
            output = model(data)
            output=F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy=100. * correct / len(test_loader.dataset)

    return test_loss, accuracy
    
def train(model, device, train_loader, optimizer):
    #model.train()
    correct=0
    train_loss=0
    batch_idx=0
    for data, target in train_loader:
        batch_idx+=1
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        output=F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        train_loss+=loss
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        #stop training early
        #if batches==batch_idx:
        #    break

    train_loss=train_loss/len(train_loader.dataset)
    accuracy=100. * correct / len(train_loader.dataset)

    return train_loss, accuracy

def student_train(model, device, train_loader, optimizer, teacher_model=None):
    #model.train()
    if teacher_model:
        teacher_model.eval()
    correct=0
    train_loss=0
    batch_idx=0
    prev=teacher_model.fc2.weight
    for data, target in train_loader:
        batch_idx+=1
        data, target = data.to(device), target.to(device)

        target=torch.squeeze(target)

        if teacher_model:
            target=teacher_model(data)
            target=F.log_softmax(target, dim=1)

        optimizer.zero_grad()
        output = model(data)
        output=F.log_softmax(output, dim=1)
        loss = F.kl_div(input=output, target=target, log_target=True, reduction='batchmean')
        train_loss+=loss
        loss.backward()
        optimizer.step()
        new_weight=model.fc2.weight
        #np.save('fc2_weight.npy', (prev-new_weight).cpu().detach().numpy())
        #np.save('grad.npy', model.fc2.weight.grad.cpu().detach().numpy())
        #import ipdb; ipdb.set_trace()

    return train_loss/len(train_loader.dataset)
