# dlp2

**참고**

- 랜덤 포레스트와 XGBoost vs. DLP
  - [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- 트랜스포머 튜토리얼
  - [MinT: Minimal Transformer Library](https://github.com/dpressel/mint)

**텐서플로우 + GPU 세팅**

텐서플로우와 GPU를 이용하도록 하는 설정을 가장 간단하게 하는 방법은 다음과 같다.

1. [Microsoft Visual C++ Redistributable](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) 설치.
2. 컴퓨터 그래픽 카드에 맞는 [Nvidia Game Ready driver](https://www.nvidia.com/Download/index.aspx?lang=en-us#) 설치.
3. [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 또는 
  [Anaconda](https://www.anaconda.com/products/distribution#Downloads) 설치.
    1. 설치 과정에서 경로(PATH) 설정을 하지 말 것.
    2. 설치 완료 후 Anaconda Powershell Prompt를 시작(Start) 메뉴를 통해 실행할 것.
4. [environment-gpu.yml](https://github.com/codingalzi/dlp2/blob/master/environment-gpu-tf210.yml) 파일 다운로드.
5. Anaconda Powershell Prompt를 이용하여 아래 명령문 실행.
    1. `conda update -y -n base conda`
    2. `conda env create -f environment-gpu.yml`
    3. `conda activate dlp2`
    4. `python -m ipykernel install --user --name=python3`
    5. `jupyter notebook`

참고: GPU를 사용할 수 없는 경우[environment.yml](https://github.com/codingalzi/dlp2/blob/master/environment.yml) 파일 다운로드.
