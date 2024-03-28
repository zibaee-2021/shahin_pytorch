"""
Implementation of mixup data augmentation method (Zhang 2017)  "to construct virtual training examples"

"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from PIL import Image


class MixUp(nn.Module):

    def _save_16_to_png(self, images):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        images_concat = torch.cat(images.split(1, 0), 3).squeeze()

        for i in range(3):  # de-normalise RGB
            images_concat[i] = images_concat[i] * std[i] + mean[i]

        images_concat = torch.clamp(images_concat, 0, 1) # ensure between 0 and 1

        im = Image.fromarray((images_concat.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        im.save("train_pt_images_mixed_up.png")
        print('train_pt_images.jpg saved.')
        print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    def mixup(self, X, y, sm=1, alf=0.2):
        """
         "For mixup, we find that αlpha ∈ [0.1, 0.4] leads to improved performance over ERM, whereas for large αlpha,
        mixup leads to underfitting."

        """
        cumulative_loss = 0
        # y1, y2 should be one-hot vectors
        for i, (x1, y1), (x2, y2) in enumerate(zip(ldr1, ldr2)):
            x1, y1, x2, y2 = x1[0].to(dev), y1[1].to(dev), x2[0].to(dev), y2[1].to(dev)

            np.random.seed(42)
            if sm == 2:
                lam = np.random.uniform(low=0.0, high=1.0)
            else:
                lam = np.random.beta(alf, alf)

            x = Variable(lam * x1 + (1. - lam) * x2)
            y = Variable(lam * y1 + (1. - lam) * y2)

            # while i >= 0 and i <=16:
            #     images

            opt.zero_grad()
            loss(net(x), y).backward()
            cumulative_loss += loss.item()
            print(f'loss.item() {loss.item()}')
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, cumulative_loss / 2000))
                running_loss = 0.0

            opt.step()
        return net, cumulative_loss
