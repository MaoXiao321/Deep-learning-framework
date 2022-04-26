"""使用自带数据集"""

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10("../dataset",  # 数据存储位置
                                         train=False,  # True表示当训练集，False作为测试集
                                         transform=dataset_transform  # 对图片做的转换
                                         )
# 把test_data放到DataLoader里面，方便后续使用
test_loader = DataLoader(dataset=test_data,
                         batch_size=64,  # 每个小块的大小
                         shuffle=True,  # 是否将数据打乱
                         num_workers=0,  # 多线程处理
                         drop_last=True  # 是否舍掉最后不足一个batchsize的batch
                         )
print(f"数据集的标签概况：{test_data.class_to_idx}")
# # 测试数据集中第一张图片及target
# img, target = test_data[0]
# print(img.shape)
# print(target)

# writer = SummaryWriter("../logs")
# for epoch in range(2):
#     step = 0
#     for data in test_loader:
#         imgs, targets = data
#         # print(imgs.shape)
#         # print(targets)
#         writer.add_images("Epoch: {}".format(epoch), imgs, step)
#         step = step + 1
#
# writer.close()
