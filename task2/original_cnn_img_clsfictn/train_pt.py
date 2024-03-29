# train script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from time import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from transformers import DeiTConfig, DeiTModel, DeiTFeatureExtractor
import multiprocessing


def set_device():
    """
    Set device: to either Cuda (GPU), MPS (Apple Silicon GPU), or CPU
    """
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using {device} device')
    return device


# this function is not used at present
def process_image(img):
    # ViT: Init feature extractor
    feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
    # feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
    # feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
    return feature_extractor(images=img, return_tensors='pt').pixel_values.squeeze()


if __name__ == '__main__':

    cpu_count_mp = multiprocessing.cpu_count()
    print(f"Number of CPU threads according to multiprocessing: {cpu_count_mp}")

    start = time()
    device = set_device()

    # # CNN: transform cifar-10 dataset
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # # ViT: transform cifar-10 dataset for 'facebook/deit-tiny-patch16-224'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # # ViT: Image transformation pipeline to apply feature extraction
    # transform = Compose([
    #     transforms.ToPILImage(),  # Convert tensor/ndarray to PIL Image.
    #     process_image,
    # ])

    # ViT: Apply transformation to CIFAR10 dataset (50,000 images):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    batch_size = 20

    # Load in Dataloader:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_count_mp)
    # List classes of interest:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create iterator of images:
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Save example images to jpg:
    im = Image.fromarray((torch.cat(images.split(1, 0), 3)
                          .squeeze() / 2 * 255 + .5 * 255)
                         .permute(1, 2, 0).numpy().astype('uint8'))
    im.save("train_pt_images_vit.jpg")
    print('train_pt_images_vit.jpg saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # # CNN: instantiate untrained model
    # net = Net()

    # ViT: instantiate untrained model
    # Init DeiT `deit-tiny-patch16-224` style configuration, (with random weights)
    config = DeiTConfig.from_pretrained('facebook/deit-tiny-patch16-224')
    net = DeiTModel(config)
    print(f'Tiny net {net}')

    config = DeiTConfig()  # empty constructor appears to be base model
    net = DeiTModel(config)
    print(f'Default net {net}')

    # access embedding layers to take a look
    embedding_layer = net.embeddings
    image = torch.randn(3, 224, 224)
    output_embedding = embedding_layer(image)
    print(output_embedding)

    net = net.to(device)

    # with open('network_pt_vit.py', 'w') as f:  # Save empty model architecture to file
    #     f.write(str(net))

    # loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # epochs = 2  # only 2 used for CNN
    epochs = 20  # 20 for ViT

    # training loop
    for epoch in range(epochs):  # loop over the dataset multiple times

        loss, running_loss = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            # data is list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.pooler_output
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print(f'loss={loss.item()} at epoch={epoch}')

        # save model
        # torch.save()

    print(f'Completed training for {epochs} epochs.')

    torch.save(net.state_dict(), '../saved_models/saved_model_vit1.pt')  # save trained model
    net.save_pretrained('saved_model_vit2')  # save trained model

    print('Trained model saved.')
    print(f'Time taken to train = {time() - start} secs')
