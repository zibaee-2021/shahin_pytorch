"""
Task 2 A depth-wise separable convolution

For the purpose of the coursework, the dataset is only split into two, training and test sets.

• Adapt the Image Classification tutorial to use a different network, VisionTransformer.
You can choose any configuration that is appropriate for this application. [5]

- TensorFlow version
- PyTorch version

• Implement a data augmentation class MixUp, using the mixup algorithm, such that: [10]
  - Inherited from the relevant classes in TensorFlow/PyTorch is recommended but not assessed.
  - The MixUp algorithm can be applied to images and labels in each training iteration.
  - Have an input flag “sampling_method” and appropriate hyperparameters for two options:
▪ sampling_method = 1: λ is sampled from a beta distribution as described in the paper.
▪ sampling_method = 2: λ is sampled uniformly from a predefined range.
▪ The algorithm should be seeded for reproducible results.
  - Visualise your implementation, by saving to a PNG file “mixup.png”, a montage of 16 images
with randomly augmented images that are about to be fed into network training.
  - Note: the intention of this task is to implement the augmentation class from scratch using
only TensorFlow/PyTorch basic API functions. Using the built-in data augmentation classes
may result in losing all relevant marks.
• Implement a task script “task.py”, under folder “task2”, completing the following: [15]
  - Train a new VisionTransformer classification network with MixUp data augmentation, for
each of the two sampling methods, with 20 epochs.
  - Save the two trained models and submit your trained models within the task folder.
  - Report the test set performance in terms of classification accuracy versus the epochs.
  - Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images with
printed messages clearly indicating the ground-truth and the predicted classes for each.
"""
import os
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torchvision.datasets as tv_datasets
import task2_helper_functions as hfun2
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

""" ######## IMPORTANT FLAG: ACTION REQUIRED ######## """
# SET THIS FLAG TO TRUE IF YOU JUST WANT TO DO INFERENCE (AND NOT DO FINE-TUNING OF
# PRETRAINED VIT MODEL):
# load_finetuned_vit_for_inference_only = True
load_finetuned_vit_for_inference_only = False
""" ################################################# """

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
    sampling_method = 1  # 1 FOR UNIFORM
    # sampling_method = 2  # 2 for BETA
    all_20_epochs_start = time()
    for epoch in tqdm(range(epochs)):
        print(f'\nEpoch#{epoch + 1}')
        # 1. FINE-TUNE PRETRAINED MODEL:
        train_loss_per_epoch, train_accuracy = hfun2.fine_tune(device, pretrained_vit, trainloader, criterion, opt,
                                                               epoch, sampling_method)
        # 2. EVALUATE ON TEST-SET AFTER EACH EPOCH OF FINE-TUNING:
        test_loss_per_epoch, test_accuracy = hfun2.test_inference(device, pretrained_vit, testloader, criterion)
        test_losses[epoch] = test_loss_per_epoch
        test_accs[epoch] = test_accuracy

    # # SAVE LOSSES & ACCURACIES FOR EACH OF 20 EPOCHS TO CSV FILES:
    train_losses_np = train_losses.cpu().numpy()
    test_losses_np = test_losses.cpu().numpy()
    test_accs_np = test_accs.cpu().numpy()
    train_accs_np = train_accs.cpu().numpy()
    losses_accs_dirs = f'saved_models/acc_losses/sm_{sampling_method}'
    if not os.path.exists(losses_accs_dirs): os.makedirs(losses_accs_dirs)
    vit_train_losses_path = os.path.join(losses_accs_dirs, 'train_losses_np.csv')
    vit_test_losses_path = os.path.join(losses_accs_dirs, 'test_losses_np.csv')
    vit_train_accs_path = os.path.join(losses_accs_dirs, 'train_accs_np.csv')
    vit_test_accs_path = os.path.join(losses_accs_dirs, 'test_accs_np.csv')

    np.savetxt(vit_test_losses_path, test_losses_np, delimiter=',')
    np.savetxt(vit_test_accs_path, test_accs_np, delimiter=',')
    np.savetxt(vit_train_losses_path, train_losses_np, delimiter=',')
    np.savetxt(vit_train_accs_path, train_accs_np, delimiter=',')

    print(f'Classification accuracy per epoch for test set= {test_losses}')
    print(f'END - Fine-tuning model for {epochs} epochs took {round(((time() - all_20_epochs_start) / 60), 4)} mins')

    """ ######## IMPORTANT DON'T CHANGE THIS FLAG ######## 
    UNLESS YOU'RE 100% SURE ABOUT OVERWRITING MY FINE-TUNED MODELS 
    OTHERWISE YOU SHOULD MANUALLY CHANGE THE FILE NAMES"""
    save_fine_tuned_model = False
    if save_fine_tuned_model:
        tuned_model_dirs = f'saved_models/pretrained_finetuned/sm_{sampling_method}'
        if not os.path.exists(tuned_model_dirs): os.makedirs(tuned_model_dirs)
        fine_tuned_path = os.path.join(tuned_model_dirs, 'vit_finetuned.pt')
        torch.save(pretrained_vit.state_dict(), fine_tuned_path)
        print('Trained model saved.')