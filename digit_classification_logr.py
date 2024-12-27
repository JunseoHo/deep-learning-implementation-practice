from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

"""

일반적인 회귀분석이 연속값을 예측하는 것이라면, 로지스틱 회귀는 범주 예측에 활용된다.

일반적인 회귀분석은 연속형 변수를 예측하며, 모형탐색방법으로 최소제곱법을 사용하고, F-테스트와 t-테스트로 모형을 검정한다.
로지스틱 회귀분석은 이산형 변수를 예측하며, 모형탐색방법으로 최대우도법을 사용하고, X2 테스트로 모형을 검정한다.

최소제곱법과 최대우도법은 임의의 표본에서 모집단 모수를 추정할 때 활용된다. (단, 표본선택편향이 발생하지 않았다고 가정한다.)
최소제곱법은 평균제곱오차와 그 공식이 동일하다. -> Σ(y_i - ŷ_i)²

최대우도법을 이해하려면 먼저 우도 개념을 알아야한다.

우도 또는 가능도는 주어진 모수에서 관측된 데이터가 나타날 확률이다.

예를 들어, 동전을 4번 던진다고 가정하자. 결과는 [앞, 앞, 뒤, 앞]이었다.
그럼 일단 '관측된 데이터'는 '[앞, 앞, 뒤, 앞]'이다. 그렇다면 동전을 4번 던졌을 때 '[앞, 앞, 뒤, 앞]'이 나올 확률이 가장 높게 만드려면
동전의 앞면이 나올 확률 θ는 얼마여야할까? 한번 손으로 계산해보자.

1. θ가 0.5라면: 0.5³ * 0.5 = 0.0625
2. θ가 0.7라면: 0.7³ * 0.3 = 0.1029
3. θ가 0.75라면: 0.75³ * 0.25 = 0.1055

더 계산해볼 수도 있겠지만 미분을 통해 직접 계산해보면 0.1055가 최대우도이며 0.75가 우도를 최대로 만드는, 동전의 앞면이 나올 확률이라는 사실을 알 수 있다.

물론 우리는 동전의 앞면이 나올 확률이 일반적으로는 0.5라는 사실을 알고 있다.
하지만 '관측된 데이터 = [앞, 앞, 뒤, 앞]'에 한해서는 0.75가 최대우도를 만들어내는 '동전의 앞면이 나올 확률'인 것이다.


"""

digits = load_digits()  # 사이킷런에서 제공하는 숫자 이미지 데이터세트

plt.figure(figsize=(20, 4)) # 일단 20 * 4 크기의 캔버스 영역을 생성한다.
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# 모델 학습
logisticRegr = LogisticRegression(max_iter=2000)    # max_iter가 너무 낮으면 STOP: TOTAL NO. of ITERATIONS REACHED LIMIT. 라는 에러가 발생한다. 즉, 학습 결과가 반복 횟수 내에서 수렴하지 않았다는 것.
logisticRegr.fit(X_train, y_train)

# 모델 테스트
print(logisticRegr.predict(X_test[0].reshape(1, -1)))
print(logisticRegr.predict(X_test[0:10]))

# 모델 성능 평가
print(f"Accuracy: {logisticRegr.score(X_test, y_test)}")


