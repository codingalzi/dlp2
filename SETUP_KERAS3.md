# Deep Learning with Python (3rd Edition) - Conda Environment Setup

이 가이드는 "Deep Learning with Python, Third Edition"의 주피터 노트북을 오류 없이 실행하기 위한 conda 환경 설정 방법을 설명합니다.

# Deep Learning with Python (3rd Edition) - Conda Environment Setup

이 가이드는 "Deep Learning with Python, Third Edition"의 주피터 노트북을 오류 없이 실행하기 위한 conda 환경 설정 방법을 설명합니다.

## 환경 파일 설명

### 1. CPU 전용 환경: `environment-keras3.yml`
- CPU에서만 실행하는 경우 사용
- Keras 3.x + 다중 백엔드 지원 (TensorFlow, PyTorch, JAX)
- 모든 시스템에서 안정적으로 작동

### 2. TensorFlow GPU 환경: `environment-keras3-simple.yml`
- TensorFlow를 주 백엔드로 사용하는 GPU 환경
- 의존성 충돌을 최소화한 단순한 구성
- **권장**: 대부분의 노트북에 적합

### 3. PyTorch 중심 환경: `environment-keras3-pytorch.yml`
- PyTorch를 주 백엔드로 사용하는 환경
- Conda에서 PyTorch와 CUDA를 관리
- PyTorch 기반 실험에 적합

### 4. 복합 GPU 환경: `environment-keras3-gpu.yml`
- 여러 백엔드를 모두 지원하려는 경우 (주의: 의존성 충돌 가능)

## 권장 설치 순서

### 1단계: 간단한 환경으로 시작 (권장)
```bash
# 가장 안정적인 환경
conda env create -f environment-keras3-simple.yml
conda activate dlp3-tf-gpu
```

### 2단계: 필요시 추가 백엔드 설치
```bash
# 환경 활성화 후 필요한 백엔드만 추가
conda activate dlp3-tf-gpu

# PyTorch 추가 (선택사항)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# JAX 추가 (선택사항)
pip install jax[cuda12]
```

## 환경별 사용법

### CPU 전용 환경
```bash
conda env create -f environment-keras3.yml
conda activate dlp3-keras3
```

### TensorFlow GPU 환경 (권장)
```bash
conda env create -f environment-keras3-simple.yml
conda activate dlp3-tf-gpu
```

### PyTorch 중심 환경
```bash
conda env create -f environment-keras3-pytorch.yml
conda activate dlp3-pytorch
```

## Keras 백엔드 설정

Keras 3는 여러 백엔드를 지원합니다. 환경 변수로 백엔드를 설정할 수 있습니다:

### TensorFlow 백엔드 (기본 권장)
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
```

### PyTorch 백엔드
```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

### JAX 백엔드
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```

## 주의사항

1. **패키지 호환성**: TensorFlow 2.17.1과 tf-keras 2.17.0은 호환되도록 설정되었습니다.

2. **GPU 환경**: GPU 환경을 사용하려면 시스템에 NVIDIA GPU와 적절한 드라이버가 설치되어 있어야 합니다.

3. **Kaggle 데이터**: 일부 노트북에서 Kaggle 데이터를 사용하므로 Kaggle 계정과 API 키가 필요할 수 있습니다.

4. **메모리 사용량**: 일부 모델은 많은 메모리를 사용하므로 시스템 사양을 확인하세요.

## 환경 테스트

환경이 올바르게 설정되었는지 확인하려면 다음 코드를 실행해보세요:

```python
# 기본 테스트
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np

print(f"Keras version: {keras.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# GPU 테스트 (GPU 환경에서만)
if tf.config.list_physical_devices('GPU'):
    print("GPU devices found:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  {gpu}")
else:
    print("No GPU devices found")

# 간단한 모델 테스트
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(8,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
print("Model created successfully!")
```

## 문제 해결

### CUDA 의존성 충돌 해결

**문제**: TensorFlow와 PyTorch가 서로 다른 CUDA 버전을 요구할 때
```
ERROR: Cannot install tensorflow[and-cuda] and torch because these package versions have conflicting dependencies.
```

**해결책 1**: 단일 백엔드 환경 사용 (권장)
```bash
# TensorFlow만 사용
conda env create -f environment-keras3-simple.yml

# 또는 PyTorch만 사용  
conda env create -f environment-keras3-pytorch.yml
```

**해결책 2**: CPU 버전으로 시작 후 필요시 GPU 추가
```bash
# CPU 환경 생성
conda env create -f environment-keras3.yml
conda activate dlp3-keras3

# GPU 지원이 필요한 경우 개별 설치
pip install tensorflow[and-cuda]==2.17.1
# 또는
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 의존성 충돌이 발생하는 경우
```bash
# 환경 삭제 후 재생성
conda env remove -n dlp3-keras3
conda env create -f environment-keras3.yml
```

### 특정 패키지 업데이트가 필요한 경우
```bash
conda activate dlp3-keras3
pip install --upgrade 패키지명
```

### GPU 인식 문제
```python
# GPU 확인
import tensorflow as tf
print("TensorFlow GPU:", tf.config.list_physical_devices('GPU'))

# PyTorch GPU 확인
import torch
print("PyTorch CUDA:", torch.cuda.is_available())
print("PyTorch GPU count:", torch.cuda.device_count())
```

## 추가 정보

- [Keras 3 공식 문서](https://keras.io/)
- [Deep Learning with Python 노트북](https://github.com/fchollet/deep-learning-with-python-notebooks)
- [TensorFlow 설치 가이드](https://www.tensorflow.org/install)
