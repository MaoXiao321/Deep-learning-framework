import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# 定义设备
device = torch.device('cuda')

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())
print(f"训练集size:{train_data.data.shape}")
print(f"测试集size:{test_data.data.shape}")
print(f"标签:{train_data.class_to_idx}")

# # 获取某张图片
# img, target = test_data[0]
# print(img.shape)
# print(target)

# 用DataLoader加载数据
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=0, drop_last=True)


# 加载预训练的模型
# vgg16_pretrained = torchvision.models.vgg16(pretrained=True)  # 原网路要分1000个类
# vgg16_pretrained.classifier[6] = nn.Linear(4096, 10)  # 修改下全连接层

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load('tudui_29_gpu.pth')
model.to(device)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

total_test_loss = 0
total_test_accuracy = 0

i = 1
for data in test_loader:
    imgs, targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)

    outputs = model(imgs)  # 预测结果
    loss = loss_fn(outputs, targets)
    total_test_loss += loss  # 测试集上的总损失
    accuracy = (outputs.argmax(1) == targets).sum()
    print(outputs.argmax(1))
    total_test_accuracy += accuracy
    i -= 1
    if i % 10 == 0:
        print(i)
    if i == 0:
        break

print("整体测试集上的Loss: {}".format(total_test_loss))
print("整体测试集上的正确率: {}".format(total_test_accuracy / 7000))

# writer = SummaryWriter('../logs')
# step = 0
# for data in test_loader:
#     imgs, targets = data  # 获取每个batch中的图片和标签
#     writer.add_images('train_loader', imgs, step)
#     step += 1
# writer.close()
