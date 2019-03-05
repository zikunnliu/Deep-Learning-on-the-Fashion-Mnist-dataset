from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import urllib.request
import numpy as np
import random
import struct
import torch
import errno
import math
import gzip
import io
import os
import sklearn.metrics


class Fashion(Dataset):
    """Dataset: https://github.com/zalandoresearch/fashion-mnist
	Args:
		root (string): Root directory of dataset where ``processed/training.pt``
			and  ``processed/test.pt`` exist.
		train (bool, optional): If True, creates dataset from ``training.pt``,
			otherwise from ``test.pt``.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
		transform (callable, optional): A function/transform that takes in a numpy image
			and may return a horizontally flipped image."""

    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    ]

    file_name = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]

    raw = "raw"
    processsed = "processsed"

    def __init__(self, root, train=True, transform=True, download=False):
        super(Fashion, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.tensor_transform = transforms.ToTensor()

        raw_path = os.path.join(self.root, self.raw)
        if download and (os.path.exists(raw_path) == False):
            self.download(self.root)

        if self.train:
            train_path = os.path.join(self.root, self.processsed, "training_set.pt")
            self.train_images, self.train_labels = torch.load(train_path)
        else:
            test_path = os.path.join(self.root, self.processsed, "testing_set.pt")
            self.test_images, self.test_labels = torch.load(test_path)

    '''
	__getitem__(index) -> Will return the image and label at the specified index
	
	If transform parametr of class is set as True the function would or would not
	perform a random horizontal flip of the image.
	'''

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_images[index], self.train_labels[index]
        else:
            image, label = self.test_images[index], self.test_labels[index]

        image = image.numpy()
        image = np.rot90(image, axes=(1, 2)).copy()

        if self.transform and self.train:
            image = self.transform_process(image)

        image = self.tensor_transform(image)
        image = image.contiguous()
        image = image.view(1, 28, 28)

        return image, label

    def __len__(self):
        if self.train:
            return (len(self.train_images))
        else:
            return (len(self.test_images))

    def transform_process(self, image):  # Would or would not return a flipped image
        self.rotate = random.getrandbits(1)
        image = np.flip(image, self.rotate).copy()
        return image

    '''
	download(root) -> The function will download and save the MNIST images in raw
	format under the 'raw' folder under the user specified root directory
	'''

    def download(self, root):
        raw_path = os.path.join(self.root, self.raw)
        processsed_path = os.path.join(self.root, self.processsed)

        try:
            os.makedirs(raw_path)
            os.makedirs(processsed_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        for file_index in range(len(self.file_name)):
            print("Downloading:", self.urls[file_index])
            urllib.request.urlretrieve(self.urls[file_index], (self.file_name[file_index] + '.gz'))
            print("Extracting:", self.file_name[file_index] + ".gz")
            f = gzip.open(self.file_name[file_index] + '.gz', 'rb')
            with open(raw_path + "/" + self.file_name[file_index], 'wb') as w:
                for line in f.readlines():
                    w.write(line)
            f.close()
            os.remove(self.file_name[file_index] + ".gz")

        print()
        print("Raw data downloaded and extracted in your specified root directory under /raw")
        print()
        self.process(self.root)

    '''
	process(root) -> Will process the raw downloaded files into a usable format
	and store them into the a 'processed' folder under user specified root
	directory.
	'''

    def process(self, root):
        raw_path = os.path.join(self.root, self.raw)
        processsed_path = os.path.join(self.root, self.processsed)

        print("Processing training data")
        train_image = self.readimg(self.root, self.file_name[0], 2051)
        train_label = self.readlab(self.root, self.file_name[1], 2049)
        train_data = (train_image, train_label)

        print("Processing testing data")
        test_image = self.readimg(self.root, self.file_name[2], 2051)
        test_label = self.readlab(self.root, self.file_name[3], 2049)
        test_data = (test_image, test_label)

        train_path = os.path.join(self.root, self.processsed, "training_set.pt")
        with open(train_path, "wb") as f:
            torch.save(train_data, f)

        test_path = os.path.join(self.root, self.processsed, "testing_set.pt")
        with open(test_path, "wb") as f:
            torch.save(test_data, f)
        print()
        print("Processed data has been stored in your specified root directory under /processsed")
        print()

    def readimg(self, root, file, magic):
        image = []
        path = os.path.join(self.root, self.raw, file)
        with open(path, 'rb') as f:
            magic_number, size, row, col = struct.unpack('>IIII', f.read(16))
            assert (magic_number == magic)
            for run in range(size * row * col):
                image.append(list(struct.unpack('B', f.read(1)))[0])
            image = np.asarray(image, dtype=np.float32)
            return (torch.from_numpy(image).view(size, 1, row, col))

    def readlab(self, root, file, magic):
        label = []
        path = os.path.join(self.root, self.raw, file)
        with open(path, 'rb') as f:
            magic_number, size = struct.unpack(">II", f.read(8))
            assert (magic_number == magic)
            for run in range(size):
                label.append(list(struct.unpack('b', f.read(1)))[0])
            label = np.asarray(label)
            return (torch.from_numpy(label))


train_dataset = Fashion(root="./FashionMNIST", train=True, transform=True, download=True)
test_dataset = Fashion(root="./FashionMNIST", train=False, transform=True, download=True)

'''
Calculation of total epochs using a defined batch size(batch_size)
and total iterations(n_ters)
'''

batch_size = 100
n_iters = 18000
epoch_size = n_iters / (len(train_dataset) / batch_size)

'''
Loading the test dataset (test_loader)
'''

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

image_dict = {0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
              4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',
              8: 'Bag', 9: 'Ankle Boot'}
for index, (images, labels) in enumerate(test_loader):
    image = images[index][0]
    label = labels[index]
    plt.imshow(image)
    plt.suptitle(image_dict[label.item()] + " - " + str(label.item()))
    if index == 5:
        break

'''
Model Details:

Two Convolutional Layers:
      - Using ReLU activation
      - Batch Normalisation
      - Uniform Xavier Weigths
      - Max Pooling
      
One Fully Connected Layer:
      - Using ReLU activation
      
One Fully Connected Layer:
      - Output Layer

'''


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        return out


'''
Instantiating the model class
'''

model = CNNModel()
if torch.cuda.is_available():
    model.cuda()

'''
Loss Function: Cross Entropy Loss
Optimizer: Stochastic Gradient Descent (Nesterov Momentum is enabled).

Variable: learning_rate -> Stores the learning rate for the optimizer function.
          moment -> Stores the momentum for the optimizer function.
'''

criterion = nn.CrossEntropyLoss()

learning_rate = 0.015
moment = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=moment, nesterov=True)

for epoch in range(300):

    '''
	Loading the training dataset after every epoch which will
	load it from the Fashion Dataset Class making a new train
	loader for every new epoch causing random flips and random
	shuffling of above exampled Fashion MNIST images. 
	'''
    print('epoch {}'.format(epoch + 1))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = 0.
    train_acc = 0.
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        labels = labels.long()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_dataset)), train_acc / (len(train_dataset))))


    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for image, label in test_loader:
        if torch.cuda.is_available():
            image = Variable(image.cuda())
            label = Variable(label.cuda())
        else:
            image = Variable(image)
            label = Variable(label)
        output = model(image)
        label = label.long()
        loss = criterion(output, label)
        loss = loss.long()
        eval_loss += loss.item()
        pred = torch.max(output, 1)[1]
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_dataset)), eval_acc / (len(test_dataset))))
    print('f1_score',
          sklearn.metrics.f1_score(y_true=label, y_pred=pred, labels=range(10), average='macro', sample_weight=None))