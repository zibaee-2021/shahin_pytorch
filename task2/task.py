"""
Task 2 A depth-wise separable convolution
"""
import os
from time import time
from tqdm import tqdm
import torch
import torchvision.datasets as tv_datasets
import task2_helper_functions as hfun2
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')


if __name__ == '__main__':
    """
    ############ IMPORTANT: ############################################################
    saved_models_t2.zip is publicly-accessible from:
    https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjzi_ucl_ac_uk/ESaA7LsuO0RFoJLe74a82aUBIHCFMgFppJ03y8X7Hh6MgA?e=QsVsFS
    'Anyone' with link can view this.
    MODELS ARE ALL IN ONEDRIVE AND MUST BE MOVED HERE IN ORDER FOR THIS FUNCTION TO WORK
    DO NOT SET FLAG TO TRUE UNLESS YOU HAVE MOVED THE MODEL INTO: 
    'SAVED_MODELS_T2/PRETRAINED_FINETUNED/SM_1'
    #####################################################################################
    # SET THIS FLAG TO TRUE IF YOU JUST WANT TO DO INFERENCE (AND NOT DO FINE-TUNING OF
    # PRETRAINED VIT MODEL):
    ##################################################################################### 
    """
    # load_finetuned_vit_for_inference_only = True
    load_finetuned_vit_for_inference_only = False

    """ CHANGE SAMPLING_METHOD TO 2 FOR UNIFORM DIST IN MIXUP """
    SAMPLING_METHOD = 1
    # SAMPLING_METHOD = 2
    """ ##################################################### """

    if load_finetuned_vit_for_inference_only:  # DO NOT FINE-TUNE. JUST DO INFERENCE.
        pretrained_vit, pretrained_transforms = hfun2.load_finetuned_vit_for_inference_only()
    else:  # FINE-TUNE A PRETRAINED VIT MODEL ON CIFAR-10. FREEZE WEIGHTS AND THEN ADD LAYER TO HEAD:
        pretrained_vit, pretrained_transforms = hfun2.load_pretrained_vit_for_finetuning()

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
        test_loss, test_acc = hfun2.test_inference(device, pretrained_vit, testloader, criterion)
    else:
        # FINE-TUNE AND EVALUATE ON TEST SET FOR 20 EPOCHS:
        print('20 epochs of fine-tuning and evalutation of prediction accuracy on test set, per epoch.')
        trainset = tv_datasets.CIFAR10(root='./data', train=True, download=True, transform=pretrained_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        opt = torch.optim.Adam(pretrained_vit.parameters(), lr=0.003)
        epochs = 20
        train_losses = torch.zeros(epochs)
        test_losses = torch.zeros(epochs)
        train_accs = torch.zeros(epochs)
        test_accs = torch.zeros(epochs)
        all_20_epochs_start = time()
        for epoch in tqdm(range(epochs)):
            print(f'\nEpoch#{epoch + 1}')
            # 1. FINE-TUNE PRETRAINED MODEL:
            train_loss_per_epoch, train_accuracy = hfun2.fine_tune(device, pretrained_vit, trainloader, criterion, opt,
                                                                   epoch, SAMPLING_METHOD)
            train_losses[epoch] = train_loss_per_epoch
            train_accs[epoch] = train_accuracy
            # 2. EVALUATE ON TEST-SET AFTER EACH EPOCH OF FINE-TUNING:
            test_loss_per_epoch, test_accuracy = hfun2.test_inference(device, pretrained_vit, testloader, criterion)
            test_losses[epoch] = test_loss_per_epoch
            test_accs[epoch] = test_accuracy

        # # SAVE LOSSES & ACCURACIES FOR EACH OF 20 EPOCHS TO CSV FILES:
        hfun2.save_loss_acc_to_csv(SAMPLING_METHOD, train_losses, test_losses, train_accs, test_accs)

        print(f'Classification accuracy per epoch for test set= {test_losses}')
        print(f'END - Fine-tuning model for {epochs} epochs took {round(((time() - all_20_epochs_start) / 60), 4)} mins')

        """ ######## IMPORTANT DON'T CHANGE THIS FLAG ######## 
        UNLESS YOU'RE 100% SURE ABOUT OVERWRITING MY FINE-TUNED MODELS 
        OTHERWISE YOU SHOULD MANUALLY CHANGE THE FILE NAMES"""
        save_fine_tuned_model = False
        if save_fine_tuned_model:
            tuned_model_dirs = f'saved_models_t2/pretrained_finetuned/sm_{SAMPLING_METHOD}'
            if not os.path.exists(tuned_model_dirs): os.makedirs(tuned_model_dirs)
            fine_tuned_path = os.path.join(tuned_model_dirs, 'vit_finetuned.pt')
            torch.save(pretrained_vit.state_dict(), fine_tuned_path)
            print('Trained model saved.')


