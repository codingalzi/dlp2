# WSL2에 GPU 지원 Tensorflow, PyTorch 설치 요령(2024년 7월 기준)

- 필수 준비 사항: 윈도우 11, WSL2, Ubuntu 22.04

- 우분투 운영체제에 직접 설치하는 방법도 동일 (Nvidia 드라이버 설치 부분 제외)

## NVIDIA cuda 드라이버 설치 (윈도우11 대상)

cuda(Compute Unified Device Architecture)는 병렬 처리를 사용하여 계산을 더 빠르게 처리할 수 있도록 설계된 강력한 컴퓨팅 플랫폼이며
그래픽 드라이버, 툴킷, 소프트웨어 개발 키트 및 응용 프로그래밍 인터페이스로 구성된다. cuda를 통해 NVIDIA 그래픽 카드에서 CPU에서 보다 훨씬 빠르게 실행되는 프로그램을 만들 수 있다. 이는 몇 개의 CPU 코어가 아닌 수천 개의 그래픽 카드 코어를 사용하여 계산을 수행할 수 있기 때문이다.

1. [Nvidia 드라이버 공식 웹 사이트 방문](https://www.nvidia.com/Download/index.aspx?lang=en-us)
1. 자신의 컴퓨터에 맞는 Nvidia 드라이버(Search) 탐색 후에 아래 이름과 같은 드라이버 설치 파일을 
    다운로드한 다음에 실행하여 설치한다.

    ```
    5xx.xx-desktop-win10-win11–64bit-international-dch-whql.exe
    ```

참고: [PyTorch](https://pytorch.org/get-started/locally/) 공식 홈페이지에서 cuda 버전이
11.8, 12.1, 12.4를 지원하는 PyTorch 라이브러리를 간단하게 설치하는 방법을 언급한다.
반면에 NVIDIA 최신 드라이버는 cuda 12.6을 포함한다. 테스트 결과 별 문제는 없어 보이지만
그래도 [cuda 12.4를 사용하는 NVIDIA 드라이버](https://www.nvidia.com/ko-kr/drivers/details/224483/)를 
설치할 것을 권장한다. 물론 자신의 그래픽 카드가 지원되는 경우에 한한다.

## WSL2 설치 및 업데이트

1. Windows PowerShell을 관리자 모드로 연다.
1. 아래 명령문 실행

   ```bash
   wsl --install
   ```

1. 최신 리눅스 커널을 다운로드 하기 위해 먼저 wsl을 업데이트 한다.

    ```bash
    wsl --update
    ```
        
1. 아래 명령문을 이용하여 설치된 wsl 버전이 5.15.90.1 이상이어야 함

    ```bash
    wsl uname -r
    ```

## 우분투 22.04 LTS 설치

두 가지 방식이 있다.

- 방식 1: Windows Powershell을 다시 관리자 모드로 연 다음 아래 명령문 활용

    ```bash
    wsl --install -d Ubuntu-22.04
    ```

- 방식 2:  MS Store에서 우분투 22.04 검색 후 설치

어떤 방식으로든 설치가 끝난 후 우분투를 실행할 때 요구되는 사용자 아이디와 패스워드를 지정하면
모든 설정이 끝난다.

우분투 설치 후 우분투 터미널에서 명령문을 실행하여 설치된 cuda 드라이버 버전을 확인한다.

```bash
nvidia-smi
```

터미널에서 출력된 내용 중에서 `cuda Version`을 확인한다. 
2024년 7월 기준으로 최소 12.4 이상으로 확인된다.

**참고:** cuda 버전 확인이 꼭 필요한지 이제는 확실하지 않다. 
이유는 Tensorflow와 PyTorch 모두 자체적으로 cuda 라이브러러리를 설치하기 때문이다.


## 파이썬 설치

파이썬 설치는 miniconda를 이용한다.
2024년 7월 기준으로 파이썬 3.12.4 버전이 함께 설치된다.

### miniconda와 파이썬, 주피터 노트북 설치

우분투 터미널에서 아래 명령문들을 차례대로 실행한다.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Tensorflow 설치

우분투 터미널에서 계속해서 아래 명령문을 실행한다.

```bash
pip install tensorflow[and-cuda]
```

설치가 완료된 후에 명령문으로 GPU가 제대로 작동하는지 확인한다.

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

다양한 경고문과 에러와 함께 최종적으로 아래와 같은 내용이 출력되면 정상적으로 작동하는 것이다.

```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

참고로 경고문은 무시해도 되며 에러 부분도 별 문제 없어 보인다.
텐서플로우를 실행할 때 경고를 보고 싶지 않다면 우분투 22.04의 `.bashrc` 파일에 아래 내용을 추가한다.

```bash
# Suppresseing Warnings of Tensorflow
export TF_CPP_MIN_LOG_LEVEL="2"
```

위와 같이 저장한 다음에 리눅스 터미널을 새로 열어 파이썬과 텐서플로우를 실생하면 경고문이 보이지 않는다.
경고문을 다시 보이게 하려면 숫자 2를 0으로 대체하면 된다.

### PyTorch 설치

[PyTorch](https://pytorch.org/get-started/locally/) 공식 홈페이지의 설명에 따라 우분투 터미널에서 아래 명령문을 실행한다.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

위 명령문은 cuda 12.4를 함께 설치한다.
언급된 PyTorch 홈페이지에서 현재 11.8, 12.1, 12.4 세 가지 cuda 버전을 지원한다.
앞서 `nvidia-smi` 명령문으로 확인된 cuda 버전이 12.4보다 높다 하더라도 문제가 없어 보이기도 한다.
이유는 PyTorch가 자체로 설치한 cuda 라이브러리를 사용하기 때문이지 않을까 한다.

설치가 완료된 후에 파이썬을 실행한 다음 아래 명령문을 이용하여 GPU의 지원 여부를 확인한다.

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

아래와 같이 cuda가 지원됨을 학인해주는 `True`와 설치된 그래픽카드 모델이 출력되면 정상적으로 작동하는 것이다.

```bash
True
'NVIDIA GeForce RTX 4070'
```

**주의사항:** 
`pip`을 이용하여 PyTorch를 설치하면 Tensorflow와 함께
설치된 cuda 라이브러리와의 충돌로 인해 Tensorflow에서 GPU가 지원되지 않을 수 있다.

### 추가 파이썬 라이브러리 설치

필요에 따라 우분투 터미널에서 `conda` 또는 `pip` 을 이용하여 추가 패키지를 설치한다.
일반적으로 텐서플로우 관련해서는 `pip`으로, 그 이외의 경우엔 `conda`를 추천한다.

데이터분석에 필요한 추가 필수 라이브러리와 `conda`를 이용한 설치방법은 다음과 같다.
참고로 numpy는 이전 과정에서 이미 설치되었다.

| 라이브러리 | 설치 명령문 | 설명 |
| :--- | :--- | :--- |
| jupyter | `conda install -y jupyter` | 주피터 노트북 |
| pandas | `conda install -y -c anaconda pandas` | 판다스 모듈 |
| scikit-learn | `conda install -y -c anaconda scikit-learn` | 사이킷런 라이브러리 |
| matplotlib | `conda install -y -c conda-forge matplotlib` | 시각화 라이브러리 |

아래 라이브러리들도 데이터 분석을 위해 추천된다.
하지만 `pip` 파이썬 라이브러리 관리자 명령문을 이용할 것을 추천한다.

| 라이브러리 | 설치 명령문 | 설명 |
| :--- | :--- | :--- |
| XGBoost | `pip install xgboost` | 그레이디언트 부스팅 라이브러리|
| seaborn | `pip install seaborn` | 시각화 추가 라이브러리 |
| openpyxl | `pip install openpyxl` | 엑셀파일 불러오기 |

케라스 모델의 구조를 시각화하려면 `pydot` 파이썬 모듈과 `graphviz`라는 프로그램이 컴퓨터에 설치되어 있어야 한다.

| 라이브러리 | 설치 명령문 | 설명 |
| :--- | :--- | :--- |
| pydot | `pip install pydot` | 신경망 모델 시각화 라이브러리 |
| graphviz | [설치요령](https://graphviz.gitlab.io/download/) | 운영체제에 따른 설치 |

graphviz 프로그램 설치: https://graphviz.gitlab.io/download/

구글 코랩에서는 기본으로 지원됨.

## 주피터 노트북(Jupyter Notebook) 실행

우분투 터미널에서 아래 명령을 실행한다. 

```bash
jupyter-notebook
```

이후에 터미널에 보여지는 많은 내용 중에서 아래와 같이 생긴 링크에 `Ctrl` 키와 함께 마우스 오른쪽 버튼을 누른다.

```bash
   http://localhost:8888/?token=c62a2b619d95a13fbc9ef35596ae5308dd99904525a67d8c
or http://127.0.0.1:8888/?token=c62a2b619d95a13fbc9ef35596ae5308dd99904525a67d8c
```  

또는 위 주소중에 하나를 복사해서 브라우저 주소창에서 실행하면 주피터 노트북 홈 화면이 보여진다.
주피터 노트북을 활용한 파이썬 프로그래밍은 인터넷 자료를 활용한다.

## Visual Studio Code 활용

WSL에 설치된 우분투와 윈도우용 Visual Studio Code를 연동하는 방법은
[Linux용 Windows 하위 시스템에서 Visual Studio Code 사용 시작](https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/wsl-vscode) 등 인터넷 자료를 활용한다.