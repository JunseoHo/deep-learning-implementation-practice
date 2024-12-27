import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

"""

결정트리는 순도를 높이고, 불순도와 불확실성은 낮추는 방향으로 학습된다. 정보이론에서는 순도가 증가하고 불확실성이 낮아지는 것을 '정보획득'이라고 표현한다.
여기서 순도란, 같은 종류의 데이터가 모여 있는 정도를 의미한다.

예를 들어, 상자에 들어있는 공 10개가 모두 파란공공이라면 이는 파란공의 순도가 100%인 것이다.
반대로 6개는 붉은색, 4개는 파란색이라면 파란공의 순도는 40%가 된다.

(참고로 불확실성은 정보이론에서, 불순도는 분류 문제에서 데이터의 혼잡도를 지칭하는 용도로 사용되는 비슷한 용어이다.)

불확실성을 계산하는 대표적인 방법 두 가지를 살펴보자.

(1) 엔트로피

엔트로피 = 0 = 불확실성 최소 = 순도 최대
엔트로피 = 0.5 = 불확실성 최대 = 순도 최소

'엔트로피 = 1'이 불확실성 최대가 아니라는 점에 유의하자.
엔트포리 계산 공식은 다음과 같다.

-Σ(p_i * log2(p_i))

(2) 지니계수

지니계수의 의미는 '원소 n개 중에서 임의로 2개를 선택했을 때 선택된 두 원소가 다른 그룹일 확률'이다.

지니계수의 계산 공식은 다음과 같다.

1 - ∑((p_i)²)

지니계수는 로그 계산을 요구하지 않으므로 엔트로피보다 계산이 빨라 결정 트리에서 자주 활용한다.

"""

df = pd.read_csv("dataset/titanic/train.csv", index_col = "PassengerId")


"""

index_col을 지정했을 때와, 그렇지 않았을 때의 차이는 다음과 같다.

# index_col 없이
df = pd.read_csv("dataset/titanic/train.csv")

# 결과
   PassengerId  Survived  Pclass  ...
0           1         0       3   ...
1           2         1       1   ...
2           3         1       3   ...

# index_col 지정
df = pd.read_csv("dataset/titanic/train.csv", index_col="PassengerId")

# 결과
            Survived  Pclass  ...
PassengerId                  
1                 0       3  ...
2                 1       1  ...
3                 1       3  ...

"""

# print(df.head())

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']] # 예측변수로 활용한 데이터만 추출
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna() # 결측치가 존재하는 행을 삭제

X = df.drop('Survived', axis=1) # axis의 기본값은 0인데 이 경우, index 이름이 'Survived'인 행을 삭제한다.
y = df['Survived']

# 모델 훈련 고고씽
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# 모델 테스트 고고씽
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted Not Survival', 'Predicted Survival'],  
    index=['True Not Survival', 'True Survival']
))

"""

결정트리 여러 개를 돌려서 가장 많이 나온 결과를 최종 결과를 채택(투표, Voting)하는 방법도 있다.
이처럼 복수의 결정트리를 앙상블하는 알고리즘을 랜덤포레스트라고 부른다.

"""
