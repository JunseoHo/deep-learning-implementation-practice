import torch

"""

view는 텐서의 크기를 변환할 때 사용된다. torch.tensor의 내장함수로 제공된다.

기본적으로 view 함수 적용 이전과 이후의 텐서는 원소의 개수가 동일해야한다.
'-1'를 넣으면 텐서가 자동으로 해당 차원의 크기를 추론하는데 이 때 방금 언급한 원소 개수의 동일 원칙을 따른다.

예를 들어 아래 예제에서는 전체 원소의 개수가 18개이다.
여기서 view 함수를 통해 (-1, 9) 크기로 변환하였다.
두번째 차원의 크기가 9이고 전체 원소의 개수가 18개가 되려면 첫번째 차원의 크기는 2가 될 수 밖에 없다.
따라서 viewed_tensor의 크기는 (2, 9)로 출력이 되는 것이다.

다른 예시로 (-1, 3)을 전달하면? 당연히 viewed_tensor의 크기는 (6, 3)이 된다.
참고로 차원의 크기를 계산할 수 없다면 (ex: (-1, 4)) 에러가 발생한다.

또한 두 개 이상의 차원은 추론할 수 없다. 즉, -1은 반드시 view 함수에 하나만 전달되어야 한다. (RuntimeError: only one dimension can be inferred)

"""


tensor = torch.randn(2, 3, 3)
viewed_tensor = tensor.view(-1, -1, 3)
print(f"{tensor}, Shape: {tensor.shape}")
print("\n => \n")
print(f"{viewed_tensor}, Shape: {viewed_tensor.shape}")