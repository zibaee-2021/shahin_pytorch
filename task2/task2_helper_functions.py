from time import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tv_transforms
import torchvision.datasets as tv_datasets
from PIL import Image


class MixUp(nn.Module):

    def augment(self, device, X, y, batch_size, sampling_method, alpha=0.2):
        """
        If sampling_method is 1: λ is sampled from a beta distribution as described in Zhang et al 2018.
        If sampling_method is 2: λ is sampled uniformly from a predefined range.

        "For mixup, we find that αlpha ∈ [0.1, 0.4] leads to improved performance over ERM,
        whereas for large αlpha, mixup leads to underfitting." Zhang et al.
        """
        np.random.seed(42)

        if sampling_method == 2:
            lambda_ = np.random.uniform(low=0.0, high=1.0)
        else:
            lambda_ = np.random.beta(alpha, alpha)

        lam = torch.tensor(lambda_, device=device)

        random_i = torch.randperm(batch_size).to(device)
        X2 = X[random_i, :, :, :]
        y2 = y[random_i]

        y = F.one_hot(y, num_classes=10) * 1.0
        y2 = F.one_hot(y2, num_classes=10) * 1.0
        new_X = (lam * X) + ((1. - lam) * X2)
        new_y = (lam * y) + ((1. - lam) * y2)

        return new_X, new_y


def visualise_16_mixup():
    batch_size = 16
    pretrained_transforms = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = tv_datasets.CIFAR10(root='./data', train=True, download=True, transform=pretrained_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    images, labels = MixUp().augment(device, X=images, y=labels, batch_size=batch_size, sampling_method=1)
    # Assuming these are correct normalisation params used in pretrained_transforms:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images_concat = torch.cat(images.split(1, 0), 3).squeeze()
    # De-normalise
    for i in range(3):  # 3 is for RGB images
        images_concat[i] = images_concat[i] * std[i] + mean[i]

    # Clamp values to keep between 0 and 1 (this may not be necessary if values are already scaled correctly)
    images_concat = torch.clamp(images_concat, 0, 1)
    # Convert to numpy array and then to PIL Image
    im = Image.fromarray((images_concat.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    im.save("_16_augmented_mixup_images.png")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels = torch.argmax(labels, dim=1)  # change back from one-hot
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


def load_finetuned_vit_for_inference_only():
    print('You are loading an already fine-tuned (CIFAR-10) pretrained ViT model for inference only.')
    pretrained_vit = torchvision.models.vit_b_16()
    pretrained_vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=10))
    # print('\nWeights before loading saved model:')
    # print(pretrained_vit.heads[0].weight.data)
    saved_model_path = 'saved_models/pretrained_finetuned/vit_finetuned.pt'
    pretrained_vit.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cuda')))
    # print('\nWeights after loading saved model:')
    # print(pretrained_vit.heads[0].weight.data)
    pretrained_transforms = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return pretrained_vit, pretrained_transforms


def load_pretrained_vit_for_finetuning():
    print("You're loading a pretrained ViT, in order to fine-tune it on CIFAR-10 over 20 epochs "
          "of a training loop. Evaluate the model on a test set at each epoch.")
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    pretrained_transforms = pretrained_vit_weights.transforms()
    # FREEZE MODEL PARAMETERS TO PERFORM TRANSFER LEARNING (I.E FINE-TUNING):
    for params in pretrained_vit.parameters():
        params.requires_grad=False
    pretrained_vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=10))
    return pretrained_vit, pretrained_transforms


def test_inference(device, pretrained_vit, testloader, criterion, epoch=None):
    test_start = time()
    pretrained_vit.eval()
    test_loss_per_epoch, test_accuracy = 0, 0

    with torch.inference_mode():
        for i, data in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            y_preds = pretrained_vit(inputs)
            # y_preds.shape is .  these are   logits.
            loss = criterion(y_preds, labels)

            test_pred_labels = y_preds.argmax(dim=1)
            test_accuracy += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            test_loss_per_epoch += loss.item()
            if epoch and i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, test_loss_per_epoch / 2000))
                test_loss_per_epoch = 0.0
    test_loss_per_epoch = test_loss_per_epoch / len(testloader)
    print(f'test_loss_per_epoch={test_loss_per_epoch}')
    test_accuracy = test_accuracy / len(testloader)
    print(f'test_accuracy={test_accuracy}')
    print(f'Test/inference took {round(((time() - test_start) / 60), 4)} mins')
    return test_loss_per_epoch, test_accuracy


def fine_tune(device, pretrained_vit, trainloader, criterion, opt, epoch, sampling_method):
    tune_start = time()
    pretrained_vit.train()
    train_loss_per_epoch, train_accuracy = 0, 0
    mix_up = MixUp()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        X, y = mix_up.augment(device=device, X=inputs, y=labels, sampling_method=sampling_method,
                              batch_size=inputs.shape[0])
        opt.zero_grad()
        y_preds = pretrained_vit(X)
        y_preds_class = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
        y = torch.argmax(y, dim=1)  # convert one-hot back to original
        train_accuracy += (y_preds_class == y).sum().item() / len(y_preds)
        loss = criterion(y_preds, y)
        loss.backward()
        opt.step()

        train_loss_per_epoch += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss_per_epoch / 2000))

    train_loss_per_epoch = train_loss_per_epoch / len(trainloader)
    print(f'train_loss_per_epoch={train_loss_per_epoch}')
    train_accuracy = train_accuracy / len(trainloader)
    print(f'train_accuracy={train_accuracy}')
    print(f'One epoch of tuning took {round(((time() - tune_start) / 60), 4)} mins')
    return train_loss_per_epoch, train_accuracy


if __name__ == '__main__':
    print('start')
    visualise_16_mixup()
