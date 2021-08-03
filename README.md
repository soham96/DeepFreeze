# DeepFreeze: Cold Boot Attack and Model Recovery on Commercial EdgeML Device

This repo contains the code for the paper "DeepFreeze: Cold Boot Attack and Model Recovery on Commercial EdgeML Device". You will need to runu the following files:

- `cifar10.py`: To train the different models on CIFAR10 data
- `mnist.py` To train the different models on MNIST data
- `recovery_cifar.py`: To corrupt the model trained on cifar and then perform recovery on it using Knowledge Distillation
- `recovery_mnist.py`: To corrupt the models trained on MNIST and then recover it using Knowledge Distillation
- `recovery_pretrained.py`: To corrupt pretrained ImageNet models (finetuned on CIFAR10) and then recover it
