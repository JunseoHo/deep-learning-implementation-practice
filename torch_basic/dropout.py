import torch.nn as nn
import torch

"""

드랍아웃은 각 노드가 p 확률로 비활성화된다.
절대 전체 노드에서 p 비율만큼이 비활성화되는 것이 아님의 유의하라.

아래 예제에서 반드시 10개의 노드 중 3개가 비활성화되는 것이 아니라, 각 노드가 30%의 확률로 비활성화되는 것이다.
즉, 극단적인 상황에서는 p=0.01이어도 모든 노드가 비활성화될 수도 있다.

아래 예시를 여러 번 실행해보며 0이 되는 노드가 3개가 아닌 경우를 직접 확인해보라.

TMI. 이처럼 결과가

"""


tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float()

dropout = nn.Dropout(p=0.3)
dropout.train() # 기본적으로 eval 모드로 되어 있다. eval 모드에서는 dropout이 적용되어도 비활성화가 수행되지 않는다.

print(f"{tensor} => {dropout(tensor)}")