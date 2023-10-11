# dlp2

## 텐서플로우 + GPU 세팅

1. (2023년 10월 기준) 최신 버전으로 설치하는 방법

    - 참고: [WSL2에 NVIDIA CUDA 드라이버 12.0, Toolkit 과 cuDNN 설치](./INSTALL.md)

1. WSL2 없이 윈도우 상에서 텐서플로우 2.1을 이용하는 방법
    1. [Microsoft Visual C++ Redistributable](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) 설치.
    2. 컴퓨터 그래픽 카드에 맞는 [Nvidia Game Ready driver](https://www.nvidia.com/Download/index.aspx?lang=en-us#) 설치.
    3. [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 또는 
      [Anaconda](https://www.anaconda.com/products/distribution#Downloads) 설치.
        1. 설치 과정에서 경로(PATH) 설정을 하지 말 것.
        2. 설치 완료 후 Anaconda Powershell Prompt를 시작(Start) 메뉴를 통해 실행할 것.
    4. [environment-gpu-tf210.yml](https://github.com/codingalzi/dlp2/blob/master/environment-gpu-tf210.yml) 파일 다운로드.
    5. Anaconda Powershell Prompt를 이용하여 아래 명령문 실행.
        1. `conda update -y -n base conda`
        2. `conda env create -f environment-gpu-tf210.yml`
        3. `conda activate dlp2`
        4. `python -m ipykernel install --user --name=python3`
        5. `jupyter notebook`

    참고: GPU를 사용할 수 없는 경우[environment.yml](https://github.com/codingalzi/dlp2/blob/master/environment.yml) 파일 다운로드.
