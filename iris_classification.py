import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

"""

이 예제는 KNN 알고리즘을 활용하여 붓꽃의 품종을 세토사, 버시컬러, 버지니카로 분류한다.
KNN 알고리즘은 신경망 기반이 아니므로 Pytorch를 사용하지는 않는다.
대신 sklearn에 정의된 모델을 사용하면 간편하게 KNN 알고리즘을 테스트해볼 수 있다.

"""

# 데이터 적재
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv("dataset/iris.data", names=names)

# 데이터세트를 예측변수와 결과변수로 분할
X = dataset.iloc[:, :-1].values # 모든 행을 사용하되 가장 마지막 열은 제외한다.
y = dataset.iloc[:, 4].values   # 모든 행을 사용하되 열은 앞에서 다섯 번재 값만 가져와 사용한다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 정규화: 특히 StandardScaler는 모든 예측 변수의 값들을 정규분포로 정규화한다.
# 이러한 과정이 없으면 거리에 기반하는 KNN에서, 단위가 큰 예측 변수에 의해 다른 예측 변수가 무시될 수 있다.
# fit 과정은 데이터의 평균과 표준편차를 계산하는 과정이며 transform은 이에 기반하여 데이터를 변환하는 과정이다.

"""

이 때 중요한 점은 훈련 데이터의 fit에 기반하여 테스트 데이터를 변환해야 한다는 점이다. 예를 들어보자.

X_train = np.array([
    [180, 80],
    [170, 70],
    [160, 60]
])

# 테스트 데이터
X_test = np.array([
    [190, 90],
    [150, 50]
])


X_train에서 190의 z점수는 '(190 - 170) / 10' 으로 2가 나온다.
반대로 X_test에서 190의 z점수는 '(190 - 170) / 20'으로 1이 나온다.

테스트 데이터에 어떤 데이터가 들어있는가와 무관하게, 190은 1이 아니라 2로 표준화되어야한다.
왜냐하면 190을 2로 표준화 하는 X_train으로 학습이 진행되기 때문이다.

따라서 반드시 학습 데이터의 fit에 기반하여 테스트 데이터를 변환해야하는 것이다.

"""

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# 모델 학습
knn = KNeighborsClassifier(n_neighbors=50) # K=50, 기본적으로 KNeighborsClassifier는 p=2인 민코프스키 거리, 즉 유클리드 거리를 사용한다.
knn.fit(X_train, y_train)

"""

참고로 민코프스키 거리는 p 값에 따라 다음과 같은 거리로 계산된다.

p=1 -> 맨해튼 거리
p=2 -> 유클리드 거리
p=INF -> 체비쇼프 거리

"""

# 모델 평가
y_pred = knn.predict(X_test)
# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

"""

참고로 정확도는 매 실행마다 달라질 수 있는데 이는 train_test_split 함수가 데이터를 무작위로 분할하기 때문이다.

"""


# 이번에는 최적의 K를 찾아보자.
num_of_k = 10
acc_array = np.zeros(num_of_k)

for k in np.arange(1, num_of_k + 1, 1): # k는 1부터 10까지의 값을 취한다.
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k - 1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print(f"정확도 {max_acc}으로 최적의 k는 {k + 1}입니다.")





