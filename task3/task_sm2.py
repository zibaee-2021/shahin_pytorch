# Student number: 23092186

from torch.utils.data import DataLoader, random_split

import os
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torchvision.datasets as tv_datasets
import task3_helper_functions as hfun3
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

if __name__ == '__main__':
    """
    IMPORTANT: ########################################################################
    MODELS ARE ALL IN ONEDRIVE AND MUST BE MOVED HERE IN ORDER FOR THIS FUNCTION TO WORK
    DO NOT SET FLAG TO TRUE UNLESS YOU HAVE MOVED THE MODEL INTO: 
    'SAVED_MODELS/PRETRAINED_FINETUNED/SM_1'
    #####################################################################################
    # SET THIS FLAG TO TRUE IF YOU JUST WANT TO DO INFERENCE (AND NOT DO FINE-TUNING OF
    # PRETRAINED VIT MODEL):
    ##################################################################################### 
    """
    # load_finetuned_vit_for_inference_only = True
    load_finetuned_vit_for_inference_only = False

    SAMPLING_METHOD = 2

    if load_finetuned_vit_for_inference_only:  # FOR INFERENCE. (NO FINE-TUNING)
        pretrained_vit, pretrained_transforms = hfun3.load_finetuned_vit_for_inference_only()
    else:  # FINE-TUNE A PRETRAINED VIT MODEL ON CIFAR-10. FREEZE WEIGHTS AND THEN ADD LAYER TO HEAD:
        pretrained_vit, pretrained_transforms = hfun3.load_pretrained_vit_for_finetuning()

    pretrained_vit.to(device)

    batch_size = 20
    testset = tv_datasets.CIFAR10(root='./data', train=False, download=True, transform=pretrained_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    print(f'Your flag repr(load_finetuned_vit_for_inference_only) '
          f'is currently={load_finetuned_vit_for_inference_only}')

    if load_finetuned_vit_for_inference_only:
        # INFERENCE (ON TEST-SET) ONLY:
        print(f'One-off inference only')
        test_loss, test_acc, test_mins = hfun3.test_inference(device, pretrained_vit, testloader, criterion)
    else:
        # FINE-TUNE & EVALUATE on 80%, EVALUATE ON 20% HOLDOUT TEST SET FOR 20 EPOCHS:
        print('20 epochs of fine-tuning and evalutation of prediction accuracy on test set, per epoch.')
        trainset = tv_datasets.CIFAR10(root='./data', train=True, download=True, transform=pretrained_transforms)
        full_dataset = torch.utils.data.ConcatDataset([trainset, testset])
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        trainset, testset = random_split(full_dataset, [train_size, test_size])
        validation_size = int(0.1 * len(trainset))
        train_size = len(trainset) - validation_size
        trainset, validationset = random_split(trainset, [train_size, validation_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        opt = torch.optim.Adam(pretrained_vit.parameters(), lr=0.003)
        epochs = 20
        train_losses = torch.zeros(epochs)
        val_losses = torch.zeros(epochs)
        test_losses = torch.zeros(epochs)
        train_accs = torch.zeros(epochs)
        val_accs = torch.zeros(epochs)
        test_accs = torch.zeros(epochs)
        train_mins = torch.zeros(epochs)
        val_mins = torch.zeros(epochs)
        test_mins = torch.zeros(epochs)
        all_20_epochs_start = time()
        for epoch in tqdm(range(epochs)):
            print(f'\nEpoch#{epoch + 1}')
            # 1. FINE-TUNE PRETRAINED MODEL:
            train_loss_per_epoch, train_accuracy, train_min = hfun3.fine_tune(device, pretrained_vit, trainloader,
                                                                              criterion, opt, epoch, SAMPLING_METHOD)
            train_losses[epoch] = train_loss_per_epoch
            train_accs[epoch] = train_accuracy
            train_mins[epoch] = train_min
            # 2. EVALUATE ON VALIDATION SET AFTER EACH EPOCH OF FINE-TUNING:
            val_loss_per_epoch, val_accuracy, val_min = hfun3.test_inference(device, pretrained_vit,
                                                                             validationloader, criterion)
            val_losses[epoch] = val_loss_per_epoch
            val_accs[epoch] = val_accuracy
            val_mins[epoch] = val_min
            # 3. EVALUATING ON TEST AT EVERY EPOCH IS NOT ASKED FOR (BUT I DON'T SEE HARM IN DOING THIS), IT'S VERY QUICK:
            # TEST-SET AFTER EACH EPOCH OF FINE-TUNING:
            test_loss_per_epoch, test_accuracy, test_min = hfun3.test_inference(device, pretrained_vit,
                                                                                 testloader, criterion)
            test_losses[epoch] = test_loss_per_epoch
            test_accs[epoch] = test_accuracy
            test_mins[epoch] = test_min
        # # SAVE LOSSES & ACCURACIES FOR EACH OF 20 EPOCHS TO CSV FILES:
        hfun3.save_loss_acc_mins_to_csv(SAMPLING_METHOD, train_losses, val_losses, test_losses,
                                        train_accs, val_accs, test_accs, train_mins, val_mins, test_mins)

        print(f'Classification accuracy per epoch for test set= {test_losses}')
        print(f'END - Fine-tuning model for {epochs} epochs took {round(((time() - all_20_epochs_start) / 60), 4)} mins')

        """ ######## IMPORTANT DON'T CHANGE THIS FLAG ######## 
        UNLESS YOU'RE 100% SURE ABOUT OVERWRITING MY FINE-TUNED MODELS 
        OTHERWISE YOU SHOULD MANUALLY CHANGE THE FILE NAMES"""
        save_fine_tuned_model = False
        if save_fine_tuned_model:
            tuned_model_dirs = f'saved_models/pretrained_finetuned/sm_{SAMPLING_METHOD}'
            if not os.path.exists(tuned_model_dirs): os.makedirs(tuned_model_dirs)
            fine_tuned_path = os.path.join(tuned_model_dirs, 'vit_finetuned.pt')
            torch.save(pretrained_vit.state_dict(), fine_tuned_path)
            print('Trained model saved.')


