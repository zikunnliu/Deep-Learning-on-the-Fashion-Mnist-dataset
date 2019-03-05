import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from wide_resnet import Wide_ResNet
from DWRN import Dense_Wide_ResNet
from torch.utils.data import Dataset, DataLoader

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
'''
parser = argparse.ArgumentParser(description='PyTorch fashion-mnist Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()
'''

# 超参数设置
EPOCH = 100  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 16     # 批处理尺寸(batch_size)
LR = 0.1        # 学习率



# 准备数据集并预处理
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])


trainset = torchvision.datasets.FashionMNIST(root='.processed/training.pt', train = True,download = True,transform=transform_train)
trainloader = DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)

testset= torchvision.datasets.FashionMNIST(root='.processed/test.pt', train = False,download = True,transform=transform_test)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


# 模型定义-ResNet
net = Dense_Wide_ResNet(28, 2, 0.3, 10).to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    best_acc = 85  # 初始化best test accuracy
    print("Start Training!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), float(correct) / float(total)))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),  float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f' % (float(correct) / float(total)))
                    acc = float(correct) / float(total)
                    # 将每次测试结果实时写入acc.txt文件中

                    f.write("EPOCH=%03d,Accuracy= %.3f" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=", EPOCH)

    print('Saving model......')
    torch.save(net.state_dict(), 'params.pkl')
    net.load_state_dict(torch.load('params.pkl'))