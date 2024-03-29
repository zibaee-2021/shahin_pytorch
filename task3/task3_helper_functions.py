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


def load_finetuned_vit_for_inference_only():
    """
    Load an already fine-tuned pretrained ViT model and return together with the transform as a tuple.
    (It was fine-tuned using sampling method 1).
    The models are all in onedrive at, set to 'Anyone' with link can view this:
    https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjzi_ucl_ac_uk/EbX-4pESTR1DqCChyXUn1SkBMadggCYeYpvtqUotyBMHoQ?e=eh5O46

    and so they must be moved in to task2 folder, so that `saved_models_t2/pretrained_finetuned/sm_1/vit_finetuned.pt`
    exists, before this function can be run.
    """
    print('You are loading an already fine-tuned (CIFAR-10) pretrained ViT model for inference only.')
    pretrained_vit = torchvision.models.vit_b_16()
    pretrained_vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=10))
    # print('\nWeights before loading saved model:')
    # print(pretrained_vit.heads[0].weight.data)
    saved_model_path = 'saved_models_t3/pretrained_finetuned/sm_1/vit_finetuned.pt'
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
        params.requires_grad = False
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
    test_min = round((time() - test_start) / 60, 4)
    print(f'Test/inference took {test_min} mins')
    return test_loss_per_epoch, test_accuracy, test_min


def fine_tune(device, pretrained_vit, trainloader, criterion, opt, epoch,
              sampling_method):
    train_start = time()
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
    train_min = round((time() - train_start) / 60, 4)
    print(f'One epoch of tuning took {train_min} mins')
    return train_loss_per_epoch, train_accuracy, train_min


def validation(device, pretrained_vit, validationloader, criterion, epoch=None):
    val_start = time()
    pretrained_vit.eval()
    val_loss_per_epoch, val_accuracy = 0, 0

    with torch.inference_mode():
        for i, data in enumerate(validationloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            y_preds = pretrained_vit(inputs)
            loss = criterion(y_preds, labels)
            val_pred_labels = y_preds.argmax(dim=1)
            val_accuracy += ((val_pred_labels == labels).sum().item()/len(val_pred_labels))
            val_loss_per_epoch += loss.item()
            if epoch and i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, val_loss_per_epoch / 2000))
                val_loss_per_epoch = 0.0
    val_loss_per_epoch = val_loss_per_epoch / len(validationloader)
    print(f'val_loss_per_epoch={val_loss_per_epoch}')
    val_accuracy = val_accuracy / len(validationloader)
    print(f'val_accuracy={val_accuracy}')
    val_min = round((time() - val_start) / 60, 4)
    print(f'Validation took {val_min} mins')
    return val_loss_per_epoch, val_accuracy, val_min


def save_loss_acc_mins_to_csv(sampling_method, train_losses, val_losses, test_losses,
                              train_accs, val_accs, test_accs, train_mins, val_mins, test_mins):
    train_losses_np = train_losses.cpu().numpy()
    val_losses_np = val_losses.cpu().numpy()
    test_losses_np = test_losses.cpu().numpy()
    train_accs_np = train_accs.cpu().numpy()
    val_accs_np = val_accs.cpu().numpy()
    test_accs_np = test_accs.cpu().numpy()
    train_mins_np = train_mins.cpu().numpy()
    val_mins_np = val_mins.cpu().numpy()
    test_mins_np = test_mins.cpu().numpy()
    losses_accs_mins_dirs = f'acc_losses_mins/sm_{sampling_method}'
    if not os.path.exists(losses_accs_mins_dirs): os.makedirs(losses_accs_mins_dirs)
    vit_train_losses_path = os.path.join(losses_accs_mins_dirs, 'train_losses_np.csv')
    vit_val_losses_path = os.path.join(losses_accs_mins_dirs, 'val_losses_np.csv')
    vit_test_losses_path = os.path.join(losses_accs_mins_dirs, 'test_losses_np.csv')
    vit_train_accs_path = os.path.join(losses_accs_mins_dirs, 'train_accs_np.csv')
    vit_val_accs_path = os.path.join(losses_accs_mins_dirs, 'val_accs_np.csv')
    vit_test_accs_path = os.path.join(losses_accs_mins_dirs, 'test_accs_np.csv')
    vit_train_mins_path = os.path.join(losses_accs_mins_dirs, 'train_mins_np.csv')
    vit_val_mins_path = os.path.join(losses_accs_mins_dirs, 'val_mins_np.csv')
    vit_test_mins_path = os.path.join(losses_accs_mins_dirs, 'test_mins_np.csv')
    np.savetxt(vit_train_losses_path, train_losses_np, delimiter=',')
    np.savetxt(vit_val_losses_path, val_losses_np, delimiter=',')
    np.savetxt(vit_test_losses_path, test_losses_np, delimiter=',')
    np.savetxt(vit_train_accs_path, train_accs_np, delimiter=',')
    np.savetxt(vit_val_accs_path, val_accs_np, delimiter=',')
    np.savetxt(vit_test_accs_path, test_accs_np, delimiter=',')
    np.savetxt(vit_train_mins_path, train_mins_np, delimiter=',')
    np.savetxt(vit_val_mins_path, val_mins_np, delimiter=',')
    np.savetxt(vit_test_mins_path, test_mins_np, delimiter=',')
    print(f'saved to {losses_accs_mins_dirs}')


if __name__ == '__main__':
    print('start')

