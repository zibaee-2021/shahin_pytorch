# Student number: 23092186

import os
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
        :param device: Device it is running on (e.g. 'cpu', 'cuda')
        :param X: input batch of images (tensor)
        :param y: labels of batch(tensor)
        :param batch_size: Expected to be 20.
        :param sampling_method: 1 or 2 for sampling lambda from beta distribution (as in paper) or uniform in [0,1].
        :param alpha: mixup parameter
                "For mixup, we find that αlpha ∈ [0.1, 0.4] leads to improved performance over ERM,
                whereas for large αlpha, mixup leads to underfitting." Zhang et al 2018.
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
    """
    Generate 16 images that have been processed by the mixup implementation written here (see MixUp class at top).
    Save to png file.
    """
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


def visualise_results():
    """
    Test prediction of 36 images
    Save to `result.png` file.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')
    pretrained_vit, pretrained_transforms = load_finetuned_vit_for_inference_only(device)
    pretrained_vit.to(device)

    batch_size = 36
    testset = tv_datasets.CIFAR10(root='./data', train=False, download=True, transform=pretrained_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    pretrained_vit.eval()

    # do inference to predict class
    with torch.inference_mode():
        for i, data in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            y_preds = pretrained_vit(inputs)
            test_pred_labels = y_preds.argmax(dim=1)
            print(f'predicted = {test_pred_labels} for ground-truth = {labels}')
            print(f'predicted ={test_pred_labels}')
            images, labels = next(dataiter)
            if i == 1:
                break

    # save to png
    # images, labels = next(dataiter)
    # images, labels = next(dataiter)
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
    im.save('result.png')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('Ground truth classes:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    print('Predicted classes:' + ' '.join('%5s' % classes[test_pred_labels[j]] for j in range(batch_size)))


"""
############ IMPORTANT: ############################################################
MODELS ARE ALL IN ONEDRIVE AND MUST BE MOVED HERE IN ORDER FOR THIS FUNCTION TO WORK
#####################################################################################
"""
def load_finetuned_vit_for_inference_only(device):
    """
    Load an already fine-tuned pretrained ViT model and return together with the transform as a tuple.
    (It was fine-tuned using sampling method 1).
    The models are all in onedrive at, set to 'Anyone' with link can view this:
    https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjzi_ucl_ac_uk/ESaA7LsuO0RFoJLe74a82aUBIHCFMgFppJ03y8X7Hh6MgA?e=QsVsFS
    and so they must be moved in to task2 folder, so that `saved_models_t2/pretrained_finetuned/sm_1/vit_finetuned.pt`
    exists, before this function can be run.
    """
    print('You are loading an already fine-tuned (CIFAR-10) pretrained ViT model for inference only.')
    pretrained_vit = torchvision.models.vit_b_16()
    pretrained_vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=10))
    # print('\nWeights before loading saved model:')
    # print(pretrained_vit.heads[0].weight.data)
    saved_model_path = 'saved_models_t2/pretrained_finetuned/sm_1/vit_finetuned.pt'
    pretrained_vit.load_state_dict(torch.load(saved_model_path, map_location=torch.device(device)))
    # print('\nWeights after loading saved model:')
    # print(pretrained_vit.heads[0].weight.data)
    pretrained_transforms = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return pretrained_vit, pretrained_transforms


def load_pretrained_vit_for_finetuning():
    """
    Load a pretrained ViT model from torchvision.models.
    Return the model together with the corresponding transform, as a tuple.
    """
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
    """
    Runs prediction on test batch of images using the given pretrained ViT model same loss function as used in
    training part. (Also used for inference-only functionality).
    :param device: Device it is running on (e.g. 'cpu', 'cuda')
    :param pretrained_vit: the pretrained ViT model, which may be not fine-tuned, in the process of being fine-tuned
    or having been fine-tuned over 20 epochs.
    :param testloader: The batch of images to input for classification.
    :param criterion: The loss function (same as that used for the training part).
    :param epoch: The current epoch, or none if just inference only.
    :return: loss and accuracy.
    """
    test_start = time()
    pretrained_vit.eval()
    test_loss_per_epoch, test_accuracy = 0, 0

    with torch.inference_mode():
        for i, data in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            y_preds = pretrained_vit(inputs)
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
    """
    Fine-tune a pretrained ViT model for one epoch.
    :param device: Device it is running on (e.g. 'cpu', 'cuda')
    :param pretrained_vit: the pretrained ViT model, at any stage of the fine-tuning 20 epochs.
    :param trainloader: The batch of images to input for classification.
    :param criterion: The loss function to use.
    :param opt: The Optimiser to use.
    :param epoch: The current epoch, or none if just inference only.
    :param sampling_method: For mixip augmentation, 1 for beta distribution, 2 for uniform.
    :return: loss and accuracy.
    """
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


def save_loss_acc_to_csv(sampling_method, train_losses, test_losses, train_accs, test_accs):
    """
    Save 20 losses and accuracies for each of the 20 epochs to csv files.
    :param sampling_method: 1 or 2 according to what the flag SAMPLING_METHOD is set.
    :param train_losses: 20 losses from training model over 20 epochs.
    :param test_losses: 20 losses from evaluating model on test set over 20 epochs.
    :param train_accs: 20 losses from training model over 20 epochs.
    :param test_accs: 20 losses evaluating model on test set over 20 epochs.
    """
    train_losses_np = train_losses.cpu().numpy()
    test_losses_np = test_losses.cpu().numpy()
    train_accs_np = train_accs.cpu().numpy()
    test_accs_np = test_accs.cpu().numpy()
    losses_accs_dirs = f'acc_losses/sm_{sampling_method}'
    if not os.path.exists(losses_accs_dirs): os.makedirs(losses_accs_dirs)
    vit_train_losses_path = os.path.join(losses_accs_dirs, 'train_losses_np.csv')
    vit_test_losses_path = os.path.join(losses_accs_dirs, 'test_losses_np.csv')
    vit_train_accs_path = os.path.join(losses_accs_dirs, 'train_accs_np.csv')
    vit_test_accs_path = os.path.join(losses_accs_dirs, 'test_accs_np.csv')
    np.savetxt(vit_train_losses_path, train_losses_np, delimiter=',')
    np.savetxt(vit_test_losses_path, test_losses_np, delimiter=',')
    np.savetxt(vit_train_accs_path, train_accs_np, delimiter=',')
    np.savetxt(vit_test_accs_path, test_accs_np, delimiter=',')
    print(f'saved to {losses_accs_dirs}')


if __name__ == '__main__':
    visualise_results()