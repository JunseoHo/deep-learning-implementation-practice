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

class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)

        """
        
        nn.Dropout은 텐서의 원소 중 p 비율만큼은 0으로 만들고 나머지는 '1 / (1 - p)'를 곱해 스케일링한다.

        예를 들어 p=0.2인 드롭아웃이 적용되는 과정을 살펴보자.

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        여기서 임의로 5와 9가 선택되어 0으로 바뀌었다.

        [1, 2, 3, 4, 0, 6, 7, 8, 0, 10]
        
        그리고 선택된 5와 9 외의 원소에는 '1 / (1 - 0.2) = 1.25'를 곱하는 것이다.

        [1.25, 2.5, 3.75, 6.0, 0, 7.5, 8.75, 10.0, 0, 12.5]  -> 드롭아웃이 적용된 최종 결과

        """

        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, input_data):

        """
        
        파이토치의 view 함수는 넘파이의 reshape와 동일한 역할을 수행한다. 즉, 텐서의 크기를 변경한다.
        이 모델은 28 * 28 (=784)크기의 이미지를 입력으로 받으므로 기본적으로 입력층 크기는 784여야만 한다.
        첫번째 크기는 -1로 지정하여 파이토치에게 맡기게 되는데 이 경우에는 배치의 크기가 된다. 

        """

        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

learning_rate = 0.001
model = FashionDNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print(model)

num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)
        
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print(f"Iteration: {count}, Loss: {loss.data}, Accuracy: {accuracy}")


