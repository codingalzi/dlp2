# WSL2에 NVIDIA CUDA 드라이버 12.0, Toolkit 과 cuDNN 설치 (2023년 10월 기준)

- [How to Install the NVIDIA CUDA Driver 12.0, Toolkit & cuDNN-8.8.1.3 on WSL2 in The Year 2023](https://medium.com/@soji4u2c/how-to-install-the-nvidia-cuda-driver-12-0-toolkit-cudnn-8-8-1-3-on-wsl2-in-year-2023-23165024dc16) 사이트의 내용을 최신 버전으로 업데이트 하였다.

- 필수 준비 사항: 윈도우 11, WSL2, Ubuntu 20.04

- Windows 11 + WSL2 + Ubuntu 22.04 대상: 
    - [Windows 11, WSL2, Ubuntu-22.04](https://qiita.com/rk01234/items/54f7b0a107377f1152f2) 내용 그대로 따라할 것
    - 하지만 최신 버전 설치 아님: cuda toolkit 11.8, cudnn 8.6, python 3.9, 텐서플로우 2.12 활용
    - cuda toolkit 12, python 11 등 최신 버전과의 작동여부는 아직 확인되지 않음.

- WSL2가 아닌 우분투 운영체제에 직접 설치하는 방법은 보다 쉬움.
    - [pip으로 Tensorflow 설치](https://www.tensorflow.org/install/pip?hl=ko) 내용을 그대로 따라하면 될 것으로 기대함.

## NVIDIA CUDA 드라이버 다운로드

CUDA(Compute Unified Device Architecture)는 병렬 처리를 사용하여 계산을 더 빠르게 처리할 수 있도록 설계된 강력한 컴퓨팅 플랫폼이며
그래픽 드라이버, 툴킷, 소프트웨어 개발 키트 및 응용 프로그래밍 인터페이스로 구성된다. CUDA를 통해 NVIDIA 그래픽 카드에서 CPU에서 보다 훨씬 빠르게 실행되는 프로그램을 만들 수 있다. 이는 몇 개의 CPU 코어가 아닌 수천 개의 그래픽 카드 코어를 사용하여 계산을 수행할 수 있기 때문이다.

1. [Nvidia 드라이버 공식 웹 사이트 방문](https://www.nvidia.com/Download/index.aspx?lang=en-us)
1. 자신의 컴퓨터에 맞는 Nvidia 드라이버(Search) 탐색 후 다운로드
1. 아래 이름과 같은 드라이버 설치 파일을 실행하여 설치

        5xx.xx-desktop-win10-win11–64bit-international-dch-whql.exe

## WSL2 설치 및 업데이트

1. Windows PowerShell을 관리자 모드로 연다.
1. 아래 명령문 실행

   ```bash
   wsl --install
   ```

1. 최신 리눅스 커널을 다운로드 하기 위해 먼저 wsl을 업데이트 한다.

    ```bash
    wsl --updte
    ```
        
1. 아래 명령문을 이용하여 설치된 wsl 버전이 5.15.90.1 이상이어야 함

    ```bash
    wsl uname -r
    ```

## 우분투 20.04 LTS 설치

두 가지 방식이 있다.

- 방식 1: Windows Powershell을 다시 관리자 모드로 연 다음 아래 명령문 활용

    ```bash
    wsl --install -d Ubuntu-20.04
    ```

- 방식 2:  MS Store에서 우분투 20.04 검색 후 설치

어떤 방식으로든 설치가 끝난 후 우분투를 실행할 때 요구되는 사용자 아이디와 패스워드를 지정하면
모든 설정이 끝난다.

## NVIDIA CUDA 패키지 저장소 추가

이제부터는 모든 명령문을 **우분투 20.04의 터미널**에서 실행한다.

1. NVIDIA 공개 키를 저장한다. NVIDIA 공개 키는 NVIDIA에서 출시한 소프트웨어 패키지를 다운로드하고 업데이트 하는 데에 필요한 암호화 키이다.

    ```bash
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    ```
1. 패키지 저장소 목록을 관리하는 `/etc/apt/sources.list` 에 NVIDIA CUDA 패키지 저장소 주소를 추가한다.

    ```bash
    sudo sh -c 'echo "deb  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
    ```
1. 우분투에서 사용 가능한 패키지들의 정보를 업데이트 한다. 
        
    ```bash
    sudo apt-get update
    ```

## NVIDIA CUDA Toolkit 12 설치

NVIDIA CUDA Toolkit 12는 CUDA 가속 애플리케이션을 개발하고 실행하기 위한 포괄적인 도구 세트를 제공한다.
이 툴킷에 CUDA 가속 라이브러리, 컴파일러, 도구, 샘플 및 문서와 같은 다양한 소프트웨어 구성 요소가 포함되어 있어서
개발자가 CUDA 가속 프로그램을 구현, 구축 및 실행하는 데에 도움을 준다.

먼저 아래 명령문을 실행하여 설치된 CUDA 드라이버 버전을 확인한다.

```bash
nvidia-smi
```

터미널에서 출력된 내용 중에서 `CUDA Version`을 확인한다. 
2023년 10월 기준으로 12.2로 확인된다.

아래 명령문을 실행한다.    

```bash
sudo apt-get --yes install cuda-toolkit-12-2 cuda-toolkit-11-7
```

가능한 CUDA Toolkit 버전은 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)에서 확인한다.

## CUDA와 cuDNN 라이브러리 설치

cuDNN(CUDA Deep Neural Network)은 GPU에서 딥러닝 작업을 가속화하도록 설계된 고성능 라이브러리이며
TensorFlow, PyTorch 및 Caffe와 같은 인기 있는 딥러닝 프레임워크와 원활하게 작동하도록 설계되었다.

구심층 신경망을 구축하고 훈련하는 데 일반적으로 사용되는 최적화된 연산들을 제공한다.
예를 들어 합성곱, 활성화 함수, 정규화, 풀링 연산을 지원한다.

1. 면저 [NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download)에서 로그인 한다.

1. 2023년 10월 기준으로 최신 NVIDA 그래픽 드라이버가 설치되었다는 전제 하여 아래 파일의 링크를 선택한다.

    ```bash
    Download cuDNN v8.9.5 (September 12th, 2023), for CUDA 12.x
    ```
1. 아래 버튼을 선택하여 지정된 파일을 다운로드 한다. (Intel 프로세서 기준)

    ```bash
    Local Installer for Ubuntu20.04 x86_64 (Deb)
    ```
1. 앞서 다운로드한 파일이 저장된 곳으로 이동한 후 아래 명령문을 실행한다.

    ```bash
    sudo dpkg -i cudnn-local-repo-ubuntu2004–8.9.5.29_1.0–1_amd64.deb
    ```
1. CUDA GPG 키를 불러온다.

    ```bash
    sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
    ```
1. 패키지 저장소 정도 업데이트

    ```bash
    sudo apt-get update
    ```
1. 런타임 라이브러리를 설치한다.  아래 명령문 실행하여 설치가능 패키지를 확인할 수도 있다. 
        
    ```bash
    apt-cache policy libcudnn8
    ```

    여기서는 아래 버전을 선택한다.
    
    ```bash
    sudo apt-get install libcudnn8=8.9.5.29-1+cuda12.2
    ```
1. 개발자 라이브러리 설치

    ```bash
    sudo apt-get install libcudnn8-dev=8.9.5.29-1+cuda12.2
    ```

## 파이썬과 텐서플로우 설치

파이썬 설치는 miniconda를 이용한다.

### miniconda와 파이썬, 주피터 노트북 설치

[tensorflow-install-march-2023](https://github.com/codingalzi/t81_558_deep_learning/blob/master/install/tensorflow-install-march-2023.ipynb)를 참고한 내용이다. 
이 사이트에서는 2023년 3월 기준으로 최신 텐서플로우를 설치하는 방법이 설명되어 있다.

1. miniconda 다운로드

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    ```

2. minoconda 설치

    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

3. jupyter notebook 설치

    ```bash
    conda install -y jupyter
    ```

### 텐서플로우 설치

[pip을 이용한 tensorflow 설치](https://www.tensorflow.org/install/pip)에서 Linux 또는 Windows WSL2 의 경우의 설명을 따르기만 하면 된다.
설치 과정은 다음과 같다.

- pip 업데이트

    ```bash
    pip install --upgrade pip
    ```

- GPU 지원 텐서플로우 설치

    ```bash
    pip install tensorflow[and-cuda]
    ```

이제 아래 명령문으로 GPU가 제대로 작동하는지 확인한다.

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

오류 없이 몇 개의 경고문과 함께 최종적으로 아래와 같은 내용이 출력되면 정상적으로 작동하는 것이다.

```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

`tensforslow` 라이브러리를 불러올 때 여러 종류의 경고(warning)가 함께 표기될 수도 있다.
하지만 다음 두 종류의 경고는 무시해도 된다.

- `tensorRT` 라이브러리가 없다는 경고: `libninfer_plugin` 관련
- NUMA 관련 경고
- CPU 관련 경고: AVX2 FMA 등등

위 경고를 보고 싶지 않다면 리눅스20.04의 `.bashrc` 파일에 아래 내용을 추가한다.

```bash
# Suppresseing Warnings of Tensorflow
export TF_CPP_MIN_LOG_LEVEL="2"
```

위와 같이 저장한 다음에 리눅스 터미널을 새로 열어 파이썬과 텐서플로우를 실생하면 경고문이 보이지 않는다.
경고문을 다시 보이게 하려면 숫자 2를 0으로 대체하면 된다.

### 추가 파이썬 패키지 설치

필요에 따라 `conda` 또는 `pip` 을 이용하여 추가 패키지를 설치한다.
일반적으로 텐서플로우 관련해서는 `pip`으로, 그 이외의 경우엔 `conda`를 이용한다.
각 패키지의 설치 방법은 `conda install pandas`와 같은 방식으로 인터넷에서 검색하여 확인한다.

데이터분석에 필요한 추가 필수 패키지는 다음과 같다.
참고로 numpy는 이전 과정에서 이미 설치된다.

| 패키지 | 설치 명령문 |
| :--- | :--- |
| pandas | `conda install -c anaconda pandas` |
| scikit-learn | `conda install -c anaconda scikit-learn` |
| matplotlib | `conda install -c conda-forge matplotlib` |
