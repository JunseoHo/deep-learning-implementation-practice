import torch

"""

참고로 torch로 cuda를 사용하고 싶다면 cuda 전용 torch를 설치해야한다. (그냥 torch를 설치하면 GPU가 있어도 사용할 수 없다.)

pip install torch --index-url https://download.pytorch.org/whl/cu121

torch.device는 함수처럼 보이지만 사실 클래스이다. (<class 'torch.device'>)
torch에는 클래스임에도 이름이 소문자로 시작하는 경우가 종종 있다. 당장 torch.tensor만 해도 그렇다.

참고로 전달하는 이름은 'device_type:device_index'의 포맷을 따른다.

device_index는 정수이며 device_type은 아래와 같은 문자열이 들어갈 수 있다.

cpu, cuda, ipu, xpu, mkldnn, opengl...

device_index를 생략하면 기본 장치(보통은 인덱스 0번 장치)가 선택되며 존재하지 않는 장치도 선언 자체는 가능하다.

단, to 함수 호출 시에는 에러가 발생한다. (예: PyTorch is not linked with support for mtia devices)

tensor는 선언 시 기본적으로 device가 cpu로 설정된다. 즉 RAM에 적재된다.
이를 다른 연산 장치의 메모리로 옮길 때 사용하는 함수가 to이다.

참고로 to 함수는 형변환에도 사용되며 device 변경과 형변환을 동시에 수행할 수도 있다.

tensor.to(torch.float64)    # 형변환만 수행
tensor.to(device=cpu, dtype=torch.float64)  # 형변환과 디바이스 이전까지 수행

"""


print(torch.cuda.is_available())    # GPU 사용 여부를 검사한다.

cpu = torch.device("cpu")
cuda = torch.device("cuda")
mtia = torch.device("mtia")

print(cpu)
print(cuda)
print(mtia)

tensor = torch.tensor([1, 2, 3])

print(f"tensor is allocated on {tensor.device}")

tensor.to(cpu)
tensor.to(cuda)
tensor.to(mtia)    # mtia가 없으면 에러가 뜬다.