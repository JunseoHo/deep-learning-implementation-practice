import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 원본 데이터 적재
dataset = pd.read_csv("dataset/car_evaluation.csv")
# print(dataset.head(5))
# print(dataset.tail(5))

# 데이터 분포 출력
fig_size = plt.rcParams['figure.figsize'] # matplotlib의 전역 설정을 담고 있는 딕셔너리인 rcParams로부터 기본 플롯 크기 가져오기
fig_size[0] = 8
fig_size[1] = 6
fig_size = plt.rcParams['figure.figsize'] = fig_size
dataset.output.value_counts().plot(kind='pie',
                                   autopct="%0.05f%%",
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'],
                                   explode=(0.05, 0.05, 0.05, 0.05))
# plt.show()

# print(dataset.dtypes)  # 기존의 데이터 dtype은 object로 출력된다.
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']
for categorical_column in categorical_columns:
    dataset[categorical_column] = dataset[categorical_column].astype("category")
# print(dataset.dtypes) # 형변환 이후 dtype은 category로 출력된다.

# 데이터 세트의 값을 숫자로 인코딩된 넘파이 배열로 반환한다.
# cat: 범주형 변수의 속성에 접근할 수 있는 접근자(Accessor)
# codes: 범주형 변수의 값들을 숫자로 인코딩하여 반환(자료형은 Pandas의 Series). 만약 범주 그대로 반환하고 싶다면 'categories'를 사용한다.
# values: 인코딩된 결과를 Pandas의 Series가 아닌 넘파이 배열로 반환.
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
# print(categorical_data[:10])    # 단, 숫자로 인코딩할 경우에는 어떤 숫자가 어떤 클래스에 대응되는지 코드만으로는 알 수 없다.

"""

Q. np.stack과 np.concatenate의 차이는?

결론부터 말하면 차원의 유지 여부이다.

먼저 concatenate의 경우, 기준이 되는 axis의 크기만 더해주면 쉽게 결과를 계산 가능하다.
예를 들어 크기가 (3, 2)인 2개의 넘파이 배열이 있다면 concatenate의 결과는 아래와 같다.

axis=0 -> (3 + 3, 2) -> (6, 2)
axis=1 -> (3, 2 + 2) -> (3, 4)

하지만 stack은 concatenate와 달리 새로운 차원을 추가한다.

예를 들어 크기가 (3, 2)인 5개의 넘파이 배열이 있다면 stack의 결과는 아래와 같다.

axis=0 -> (5, 3, 2)
axis=1 -> (3, 5, 2)
axis=2 -> (3, 2, 5)

즉, stack에서 axis는 새로운 차원이 추가될 위치를 나타내며 해당 차원의 크기는 스태킹하는 배열의 개수가 된다.

결과에서 보면 알겠지만 concatenate는 기준이 되는 차원은 크기가 달라도 되는 반면, stack은 반드시 모든 배열의 차원이 동일해야한다.

"""

# 예측 변수를 텐서로 변환.
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
# print(categorical_data[:10]) # 이제 배열이 아닌 tensor로 출력된다.

# 결과 변수를 텐서로 변환.
outputs = pd.get_dummies(dataset.output) # 범주형 변수를 더미 변수로 변환한다. 이 예제의 경우, 결과 변수의 범주 개수가 5개이므로 4개의 가변수로 변환된다.
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

# 각 카테고리의 고유값이 몇 개인지 추출.
categorical_column_size = [len(dataset[categorical_column].cat.categories) for categorical_column in categorical_columns]
# 각 카테고리의 임베딩 차원 수를 계산. (일반적으로 고유값의 개수를 2로 나눈 값을 사용한다. 단, 차원이 너무 많아지는 것을 방지하기 위해 min 함수 사용.)
categorical_embedding_sizes = [(column_size, min(50, (column_size + 1) // 2)) for column_size in categorical_column_size]

total_records = len(categorical_data)
test_records = int(total_records * 0.2) # 전체 데이터 중 20%를 테스트 데이터로 사용한다.

# 전체 데이터를 훈련 데이터와 테스트 데이터로 분할.
categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records:total_records]

# 모델 구현현
class Model(nn.Module): # 클래스로 구현되는 모델은 nn.Module을 상속 받는다.
    # __init__은 모델에서 사용할 파라미터와 신경망 초기화를 위해 선언한다.
    def __init__(self, embedding_size, output_size, layers, p=0.4): # p는 드롭아웃 값이며 기본 값은 0.5이다.
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        num_categorical_columns = sum((nf for ni, nf in embedding_size))

        input_size = num_categorical_columns

        for i in layers:
            all_layers.append(nn.Linear(input_size, i)) #'y = wx + b' 라는 선형변환으로 계산되므로 이런 이름이 붙었다.
            all_layers.append(nn.ReLU(inplace=True))    # 활성화 함수
            all_layers.append(nn.BatchNorm1d(i))    # 배치 정규화
            all_layers.append(nn.Dropout(p))    # 드롭아웃: p 비율만큼의 임의의 연결을 비활성화한다. 과적합 방지를 위해 사용.
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x

model = Model(categorical_embedding_sizes, 4, [200, 100, 50], p=0.4)
# print(model) # 모델의 아키텍처를 출력한다.

# 손실함수와 옵티마이저 선언.
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 연산 장치 설정.
if torch.cuda.is_available():
    device = torch.device("cuda")   # GPU가 있다면 GPU를 사용.
else:
    device = torch.device("cpu")

# 초매개변수 선언.
epochs = 500
aggregated_losses = []
train_outputs = train_outputs.to(device=device, dtype=torch.int64)

# 모델 학습 시작
for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data).to(device)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i % 25 == 1:
        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

# 모델 테스트
test_outputs = test_outputs.to(device=device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f"Loss: {loss:.8f}")
