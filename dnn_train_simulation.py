import numpy as np

"""

교재에는 없지만 파라미터가 학습되는 과정을 직접 눈으로 확인해보고 싶어 만든 시뮬레이션.

XOR 문제를 해결하는 간단한 딥러닝 모델을 구현하고 학습 과정을 시각화 해보고자 한다.


"""

testcases = [    # [입력, 입력, 출력]
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 입력층 뉴런 (입력값, 기울기)
il0 = [0, 0]
il1 = [0, 0]

# 은닉층 뉴런
hl0 = [0, 0]
hl1 = [0, 0]

# 출력층 뉴런
ol = [0, 0]

# 입력층 -> 은닉층 가중치 (가중치 값, 바이어스)
w_il0_hl0 = [np.random.random(), np.random.random()]
w_il0_hl1 = [np.random.random(), np.random.random()]
w_il1_hl0 = [np.random.random(), np.random.random()]
w_il1_hl1 = [np.random.random(), np.random.random()]

# 은닉층 -> 출력층 가중치
w_hl0_ol = [np.random.random(), np.random.random()]
w_hl1_ol = [np.random.random(), np.random.random()]

def leaky_relu(x, alpha=0.01):
   return np.maximum(alpha * x, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(x0, x1):
    il0[0] = x0
    il1[0] = x1

    hl0[0] = (il0[0] * w_il0_hl0[0] + w_il0_hl0[1]) + (il1[0] * w_il1_hl0[0] + w_il1_hl0[1])
    hl1[0] = (il0[0] * w_il0_hl1[0] + w_il0_hl1[1]) + (il1[0] * w_il1_hl1[0] + w_il1_hl1[1])

    hl0[0] = leaky_relu(hl0[0])
    hl1[0] = leaky_relu(hl1[0])

    ol[0] = (hl0[0] * w_hl0_ol[0] + w_hl0_ol[1]) + (hl1[0] * w_hl1_ol[0] + w_hl1_ol[1])

    ol[0] = sigmoid(ol[0])

    return ol[0]

def backward():
    pass

for epoch in range(0, 100):
    for testcase in testcases:
        confidence = forward(testcase[0], testcase[1])
        loss = -(testcase[2] * np.log(confidence) + (1-testcase[2]) * np.log(1-confidence))
        

# for test in data:
#     y = forward(test[0], test[1])
#     y = 1 if y >= 0.5 else 0
#     print(test, y == test[2])