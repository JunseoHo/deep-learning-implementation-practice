import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms # 데이터 전처리를 위해 사용한다
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


"""

fashion_mnist 데이터세트는 0부터 255의 값을 원소로 가지는 28 * 28 크기의 넘파이 배열이다. 즉, 이미지 데이터이다.
라벨은 0부터 9까지의 수로 이루어지는데 각각 운동화, 셔츠와 같은 의류에 대응된다.

이 예제에서는 CNN을 활용하여 이미지를 분류하는 딥러닝 모델을 구축한다.

"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 연산 장치 설정

# 데이터 다운로드
dataset_dir = "./dataset"
train_dataset = torchvision.datasets.FashionMNIST(dataset_dir, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST(dataset_dir, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

# 데이터로더 선언
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)   # 100개 단위로 묶는다.
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# 이미지 확인
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
# plt.show()
