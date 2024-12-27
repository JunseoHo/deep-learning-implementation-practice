from torch import nn
import torch

"""

평균 제곱 오차는 실측치와 예측치의 차이를 제곱하여 평균 낸 것이다.
주로 회귀 문제에서 사용된다.

"""

y1 = torch.tensor([0, 1, 2, 3, 4]).float()
y2 = torch.tensor([4, 3, 2, 1, 0]).float()
y3= torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1], requires_grad=True).float()
y4 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True).float()


print(f"y1={y1}")
print(f"y2={y2}")
print(f"y3={y3}")
print(f"y4={y4}")

print("### MSE ###")
mse = nn.MSELoss(reduction="mean")  # reduction을 sum으로 지정할 수도 있다.
print(mse(y1, y2))


"""

크로스 엔트로피 오차는 분류 문제에서 원핫 인코딩을 활용했을 때만 사용 가능한 함수이다.

torch에서는 BCELoss, CrossEntropyLoss와 같은 함수를 제공하지만 모두 내부적으로 별도의 전처리가 수행되어 순수한 크로스 엔트로피로 계산되지는 않는다.

"""

print("### CEE ###")
output = -torch.sum(y4 * torch.log(y3)) # 전처리 없이 순수하게 크로스 엔트로피만 계산
print(output)

