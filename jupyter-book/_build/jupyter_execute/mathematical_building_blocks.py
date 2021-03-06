#!/usr/bin/env python
# coding: utf-8

# (ch:building_blocks)=
# # 신경망 구성 요소

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 신경망 구성 요소](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB_mathematical_building_blocks.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# 아래 요소들을 직관적으로 살펴본다.
# 
# - 텐서(tensor)
# - 텐서 연산
# - 경사 하강법
# - 역전파

# ## 신경망 모델 활용법 소개

# MNIST 손글씨 데이터셋을 대상으로 분류 신경망 모델을 훈련시키고 활용하는 방법을
# 간단하게 소개한다.

# **케라스로 MNIST 데이터셋 불러오기**
# 
# ```python
# from tensorflow.keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# ```

# - 손글씨 숫자 인식 용도 데이터셋. 28x28 픽셀 크기의 이미지 70,000개의 샘플로 구성
#     레이블: 0부터 9까지 10개의 클래스 중 하나
# - 훈련셋: 샘플 60,000개 (모델 학습용)
#     - `train_images`
#     - `train_labels`
# - 테스트셋: 샘플 10,000개 (학습된 모델 성능 테스트용)
#     - `test_images`
#     - `test_labels`

# <div align="center"><img src="https://miro.medium.com/max/1313/1*Ow-sTZt40xg3YbyWJXNQcg.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://towardsdatascience.com/exploring-how-neural-networks-work-and-making-them-interactive-ed67adbf9283">Towards data science: Mikkel Duif(2019)</a>&gt;</div></p>

# :::{admonition} 샘플, 타깃, 레이블, 예측값, 클래스
# :class: info
# 
# 머신러닝 모델 학습에 사용되는 데이터셋과 관련된 기본 용어는 다음과 같다.
# 
# - 샘플<font size='2'>sample</font>: 개별 데이터
# - 타깃<font size='2'>target</font>: 개별 샘플과 연관된 값. 모델이 맞춰야 하는 값
# - 레이블<font size='2'>label</font>: 분류 과제의 경우 타깃 대신에 레이블 용어 사용
# - 예측값<font size='2'>prediction</font>: 개별 샘플에 대해 모델이 예측한 값
# - 클래스<font size='2'>class</font>: 분류 모델이 예측할 수 있는 레이블들의 집합. 범주<font size='2'>category</font>라고도 함. 파이썬 프로그래밍 언어의 클래스 개념과 다름에 주의할 것.
# :::

# **신경망 모델의 구조 지정**
# 
# MNIST 분류 모델로 다음 신경망을 사용한다.
# 
# ```python
# from tensorflow import keras
# from tensorflow.keras import layers
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# ```
# 
# 위 신경망 구조에 사용된 요소들은 다음과 같다.
# 
# - 층<font size='2'>layer</font>
#     - 2개의 `Dense` 층 사용. 다른 종류의 층도 사용 가능.
#     - 입력 데이터 변환 후 이어지는 층으로 전달
# - `Sequential` (자료형) 클래스 활용
#     - 층을 연결하는 방식 지정. 다른 층 연결 방식 클래스도 사용 가능.
#     - 완전 연결(fully connected). 조밀(densely)하게 연결되었다고 함.
# - 첫째 층
#     - 512개의 유닛<font size='2'>unit</font> 사용. 즉 512개의 특성으로 구성된 데이터로 변환
#     - 활성화 함수<font size='2'>activation function</font>: 렐루<font size='2'>Relu</font> 함수
# - 둘째 층
#     - 10개의 유닛 사용. 10개의 범주를 대상으로 해당 범부에 속할 확률 계산.
#     - 활성화 함수: 소프트맥스<font size='2'>Softmax</font> 함수. 모든 확률의 합이 1이 되도록 함.    

# **신경망 모델 컴파일**
# 
# 선언된 신경망을 학습이 가능한 모델로 만들기 위해
# 옵티마이저, 손실 함수, 성능 평가 지표를 설정하는 컴파일 과정을 실행한다.
# 
# ```python
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# ```
# 
# - `optimizer`: 
#     모델의 성능을 향상시키는 방향으로 가중치를 업데이트하는 알고리즘 지정.
#     옵티마이저라 불리며 경사하강법, 역전파 업무를 처리함.
# - `loss`: **손실 함수**<font size='2'>loss function</font> 지정.
#     학습 중에 있는 모델의 성능을 손실값으로 측정. 손실값이 작을 수록 좋음.
# - `metrics`: 
#     훈련과 테스트 과정을 모니터링 할 때 사용되는 평가 지표<font size='2'>metric</font> 지정.
#     손실 함수값을 사용할 수도 있고 아닐 수도 있음.
#     여러 개의 지표를 사용할 수 있지만 분류 모델의 경우 일반적으로 정확도<font size='2'>accuracy</font> 활용.

# **이미지 데이터 전처리**
# 
# 모델 학습에 좋은 방식으로 데이터를 변환하는 과정이다. 
# MNIST 데이터의 경우 
# 0부터 255 사이의 8비트 정수(`uint8`)로 이루어진 `(28, 28)` 모양의 2차원 어레이로 표현된 이미지를
# 0부터 1 사이의 32비트 부동소수점(`float32`)으로 이루어진 `(28*28, )` 모양의 1차원 어레이로 변환한다.
# 
# ```python
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype("float32") / 255   # 0과 1사이의 값
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype("float32") / 255     # 0과 1사이의 값
# ```
# 
# 전처리된 데이터가 신경망에 전달되는 과정을 묘사하면 다음과 같다.

# <div align="center"><img src="https://github.com/codingalzi/dlp2/blob/master/jupyter-book/imgs/ch02-mnist_2layers_arch.png?raw=true" style="width:600px;"></div>

# **모델 훈련**
# 
# 모델 훈련은 컴파일된 모델의 `fit()` 메소드를 호출하면 된다.
# MNIST 모델의 경우 지도 학습이기에 입력값과 타깃값을 각각 첫째와 둘째 인자로 사용한다.
# 
# ```python
# model.fit(train_images, train_labels, epochs=5, batch_size=128)
# ```
# 
# - 첫째 인자: 훈련 데이터셋
# - 둘째 인자: 훈련 레이블셋
# - `epoths`: 에포크. 전체 훈련 세트 대상 반복 훈련 횟수.
# - `batch_size`: 배치 크기. 배치 크기만큼의 훈련 데이터로 학습할 때 마다 가중치 업데이트.
# 
# 모델의 학습과정 동안 에포크가 끝날 때마다
# 평균 손실값과 평균 정확도를 계산하여 다음과 같이 출력한다.
# 
# ```
# Epoch 1/5
# 469/469 [==============================] - 5s 4ms/step - loss: 0.2551 - accuracy: 0.9263
# Epoch 2/5
# 469/469 [==============================] - 2s 4ms/step - loss: 0.1044 - accuracy: 0.9693
# Epoch 3/5
# 469/469 [==============================] - 2s 3ms/step - loss: 0.0683 - accuracy: 0.9793
# Epoch 4/5
# 469/469 [==============================] - 2s 4ms/step - loss: 0.0504 - accuracy: 0.9847
# Epoch 5/5
# 469/469 [==============================] - 2s 3ms/step - loss: 0.0378 - accuracy: 0.9885
# ```

# :::{admonition} 배치 크기와 스텝
# :class: info
# 
# **스텝**<font size='2'>step</font>은 하나의 배치(묶음)을 학습하는 과정을 의미한다.
# 배치 크기(`batch_size`)를 128로 정하면 총 6만개의 훈련 샘플을 128개씩 묶어
# 총 469(60,000/128 = 468.75)개의 배치가 매 에포크마다 생성된다.
# 따라서 에포크 한 번 동안 총 469번의 스텝이 실행되고 그럴 때마다 손실값과 정확도가 계산되며
# 이를 평균해서 훈련 과정중에 보여지게 된다.
# 위 훈련의 경우 학습된 모델의 훈련셋에 대한 정확도는 98.9% 정도로 계산되었다.
# :::

# **모델 활용: 예측하기**
# 
# 훈련에 사용되지 않은 손글씨 숫자 이미지 10장에 대한 학습된 모델의 예측값을
# `predict()` 메서드로 확인한다.
# 
# ```python
# test_digits = test_images[0:10]
# predictions = model.predict(test_digits)
# ```

# 각 이미지에 대한 예측값은 이미지가 각 범주에 속할 확률을 갖는 
# 길이가 10인 1차원 어레이로 계산된다.
# 첫째 이미지에 대한 예측값은 다음과 같다.

# ```python
# >>> predictions[0]
# array([5.6115879e-10, 6.5201892e-11, 3.8620074e-06, 2.0421362e-04,
#        2.3715735e-13, 1.0822280e-08, 3.6126845e-15, 9.9979085e-01,
#        2.0998414e-08, 1.0214288e-06], dtype=float32)
# ```

# 7번 인덱스의 값이 0.998 정도로 가장 높으며, 이는
# 0번 이미지가 숫자 7을 담고 있을 확률이 거의 100% 라고 예측함을 보여준다.
# 실제로도 0번 이미지는 숫자 7을 담고 있어서 이 경우는 정확하게 예측되었다.

# **모델 성능 테스트**
# 
# 훈련에 사용되지 않은 테스트셋 전체에 대한 성능 평가를 위해 
# `evaluate()` 메서드를 테스트 셋과 테스트 셋의 레이블셋을 인자로 해서 호출한다.
# 
# ```python
# >>> test_loss, test_acc = model.evaluate(test_images, test_labels)
# >>> print(f"test_acc: {test_acc}")
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0635 - accuracy: 0.9811
# test_acc: 0.9811000227928162
# ```
# 
# 반환값으로 손실값과 앞서 모델을 컴파일할 때 지정한 정확도가 계산된다.
# 훈련 과정과 동일하게 스텝마다 계산된 손실값과 정확도의 평균값이 출력된다.
# `evaluate()` 메서드에 사용되는 배치 크기는 32가 기본값으로 사용되기에
# 총 313(10,000/32=312.5)번의 스텝이 진행되었다.
# 
# 테스트 세트에 대한 정확도는 98% 정도이며 훈련 세트에 대한 정확도 보다 낮다.
# 이는 모델이 훈련 세트에 **과대 적합**<font size='2'>overfitting</font> 되었음을 의미한다. 
# 과대적합에 대해서는 나중에 보다 자세히 다룰 것이다.

# ## 텐서

# MNIST 손글씨 데이터 분류 모델 학습에 사용된 훈련셋과 테스트셋이 넘파이 어레이,
# 즉 `np.ndarray`로 저장되어 사용되었다. 
# 머신러닝에 사용되는 데이터셋은 일반적으로 넘파이 어레이와 같은 
# **텐서**<font size='2'>tensor</font>에 저장된다.

# **넘파이 어레이와 텐서**
# 
# 텐서는 일반적으로 숫자 데이터를 담은 모음 자료형을 가리키며
# 넘파이 어레이, 판다스 데이터프레임 등이 대표적인 텐서 자료형으로 사용된다.
# 텐서플로우 팩키지에서 자체적으로 `Tensor` 자료형을 제공하며
# 필요에 따라 내부적으로 넘파이 어레이 등을 `Tensor` 자료형으로 변환하여 처리한다.
# 
# `tf.Tensor` 는 넘파이 어레이와 매우 유사하며 다차원 어레이를 지원한다.
# 넘파이 어레이와의 차이점은 `tf.Tensor`는 GPU를 활용한 연산을 지원하지만 넘파이 어레이는 그렇지 않다.
# 또한 `tf.Tensor`는 한 번 지정하면 수정이 불가능한 불변 자료형이다. 
# 
# `tf.Tensor`와 `np.ndarray` 사이의 형변환은 필요에 따라 자동으로 이루어지기에 특별한 경우가 아니라면
# 사용자에게 편한 자료형을 사용하면 된다. 
# 여기서는 기본적으로 넘파이 어레이를 텐서로 사용한다.

# **텐서의 차원**
# 
# 텐서의 **차원**은 텐서의 표현에 사용된 **축**<font size='2'>axis</font>의 수로 
# 결정되며 **랭크**<font size='2'>rank</font>라 불리기도 한다.

# - 0차원(0D) 텐서: 정수 한 개, 부동소수점 한 개 등 하나의 수를 표현하는 텐서. 
#     일반적으로 **스칼라**<font size='2'>scalar</font>라고 불림.
#     ```
#     np.array(12)
#     np.array(1.34)
#     ```

# - 1차원(1D) 텐서: 수로 이루어진 리스트 형식. 
#     일반적으로 **벡터**<font size='2'>vector</font>로 불림.
#     한 개의 축을 가짐    
#     ```
#     np.array([12, 3, 6, 14, 7])
#     ```

# - 2차원(2D) 텐서: 행<font size='2'>row</font>과 열<font size='2'>column</font> 
#     두 개의 축을 가짐. 
#     일반적으로 **행렬**<font size='2'>matrix</font>로 불림.
#     ```
#     np.array([[5, 78, 2, 34, 0],
#               [6, 79, 3, 35, 1],
#               [7, 80, 4, 36, 2]])
#     ```

# - 3차원(3D) 텐서
#     - 행, 열, 깊이 세 개의 축 사용.
#     컬러 이미지 데이터 표현 등에 사용.
#     ```
#     np.array([[[5, 78, 2, 34, 0],
#                  [6, 79, 3, 35, 1],
#                  [7, 80, 4, 36, 2]],
#                 [[5, 78, 2, 34, 0],
#                  [6, 79, 3, 35, 1],
#                  [7, 80, 4, 36, 2]],
#                 [[5, 78, 2, 34, 0],
#                  [6, 79, 3, 35, 1],
#                  [7, 80, 4, 36, 2]]])
#     ```

# 4D 텐서는 3D 텐서로 이루어진 벡터, 5D 텐서는 4D 텐서로 이루어진 벡터 등등
# 임의의 차원의 텐서를 정의할 수 있지만 딥러닝에서는 일반적으로 4D 텐서 정도까지 사용한다.

# **텐서 속성**
# 
# 텐서의 주요 속성 세 가지는 다음과 같으며, 넘파이 어레이의 경우와 
# 동일한 기능을 갖는다.
# 
# - `ndim` 속성: 차원 수(랭크) 저장. 
#     예를 들어 MNIST 훈련셋 어레이의 차원은 3.
#     ```python
#     >>> train_images.ndim 
#     3
#     ```

# - `shape` 속성: 튜플로 저장된 축 별 크기.
#     예를 들어 MNIST의 훈련셋은 3개의 축으로 구성됨.
#     0번 축은 6만개의 샘플 데이터를,
#     1번 축은 각 이미지에 사용된 28개의 세로 픽셀 데이터를
#     2번 축은 각 이미지에 사용된 28개의 가로 픽셀 데이터를
#     담음.
#     ```python
#     >>> train_images.shape
#     (60000, 28, 28)
#     ```

# - `dtype` 속성: 항목의 자료형.
#     `float16`, `float32`,`float64`, `int8`, `uint8`, `string` 등이 
#     가장 많이 사용됨.
#     예를 들어, MNIST 훈련셋에 포함된 이미지의 픽셀 정보는 0과 255 사이의
#     정수로 표현되며 따라서 `unit8` 자료형을 사용.
#     ```python
#     >>> train_images.dtype
#     uint8
#     ```

# **텐서 활용**
# 
# 넘파이 어레이의 인덱싱, 슬라이싱 등과 동일한 기능을 이용하여
# 샘플 확인, 배치 묶음 등을 처리할 수 있다.
# 
# 예를 들어, 4번 인덱스의 이미지, 즉 5번째 이미지를 다음처럼 선택하여 확인할 수 있다.
# ```python
# >>> import matplotlib.pyplot as plt
# >>> digit = train_images[4]
# >>> plt.imshow(digit, cmap=plt.cm.binary)
# >>> plt.show()
# ```
# 
# <img src="https://github.com/codingalzi/dlp2/blob/master/jupyter-book/imgs/ch02-mnist4.png?raw=true" style="width:250px;">

# 128번 인덱스의 이미지부터 128개의 이미지로 구성된 배치는 슬라이싱을
# 이용하여 얻는다.
# ```python
# >>> batch = train_images[128:256]
# ```

# **2D 텐서 실전 예제: 벡터 데이터 활용**
# 
# 각 샘플이 여러 개의 특성값으로 이러워진 벡터로 표현되며
# 전체 데이터셋은 `(샘플 수, 특성 수)` 모양의 2D 텐서로 저장된다.
# 
# - 예제 1
#     - 샘플: 나이, 우편 번호, 소득 세 개의 특성으로 구성된 인구 통계 데이터.
#         `(3,)` 모양의 벡터로 표현.
#     - 데이터셋: 10만 명의 통계 데이터를 포함한 데이터셋은 `(100000, 3)` 모양의 2D 텐서로 표현.
# - 예제 2
#     - 샘플: 특정 문서에서 2만 개의 단어 각각이 사용된 빈도수로 구성된 데이터.
#         `(20000,)` 모양의 벡터로 표현.
#     - 데이터셋: 500개의 문서를 대상으로 할 경우 `(500, 20000)` 모양의 2D 텐서로 표현.
# - 예제 3: 사이킷런 모델의 입력 데이터셋은 기본적으로 2D 텐서임.
#     캘리포니아 주택 데이터셋, 붓꽃 데이터셋 등등.

# **3D 텐서 실전 예제: 시계열 또는 순차 데이터 활용**
# 
# 증시 데이터 등의 시계열 데이터와 트윗 데이터 등의 순차 데이터를 다룰 때 사용하며
# `(샘플 수, 타임 스텝 수, 특성 수)` 모양의 3D 텐서로 표현된다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch02-timeseries_data.png" style="width:350px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# - 예제 1
#     - 샘플: 1분마다 하루 총 390번 (현재 증시가, 지난 1분 동안 최고가, 지난 1분 동안 최저가)를 
#         측정한 데이터. `(390, 3)` 모양의 2D 텐서로 표현.        
#     - 데이터셋: 250일 동안 측정한 데이터셋은 `(250, 390, 3)` 모양의 3D 텐서로 표현.
# 
# - 예제 2
#     - 샘플: 하나의 트윗은 최대 280개의 문자로 구성되며 문자는 총 128 종류일 때
#         트윗 샘플 하나를 `(280, 128)` 모양의 2D 텐서로 표현.
#         각 항목은 0 또는 1.
#     - 데이터셋: 백만 개의 트윗은 `(1000000, 280, 128)` 모양의 3D 텐서로 표현.
#     
# 흑백 이미지로 구성된 데이터셋도 3D로 표현된다.
# 
# - 예제 3
#     - 샘플: `28x28` 크기의 (흑백) 손글씨 이미지.
#         `(28, 28)` 모양의 2D 텐서로 표현.
#     - MNIST 훈련 데이터셋: 총 6만개의 (흑백) 손글씨 이미지로 구성되.
#         `(60000, 28, 28)` 모양의 3D 텐서로 표현.

# **4D 텐서 실전 활용 예제: 컬러 이미지 데이터 활용**
# 
# 한 장의 컬러 이미지 샘플은 일반적으로 
# `(높이, 너비, 컬러 채널 수)` 또는 `(컬러 채널 수, 높이, 너비)`
# 모양의 3D 텐서로 표현한다. 
# 따라서 컬러 이미지로 구성된 데이터셋은 
# `(샘플 수, 높이, 너비, 컬러 채널 수)` 또는 `(샘플 수, 컬러 채널 수, 높이, 너비)`
# 모양의 4D 텐서로 표현된다.
# 
# RGB를 사용하는 컬러 어미지는 3개의 커널을,
# 흑백 사진은 1개의 커널을 갖는다. 
# 예를 들어 `256x256` 크기의 컬러 이미지 128개를 갖는 데이터셋 또는 배치는
# `(128, 256, 256, 3)` 모양 4D 텐서로 표현된다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch02-image_data.png" style="width:350px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 반면에 `28x28` 크기의 흑백 이미지 128개를 갖는 데이터셋 또는 배치는
# `(128, 28, 28, 1)` 모양 4D 텐서로 표현된다.
# 하지만 MNIST의 경우처럼 흑백 이미지 데이터셋은 `(128, 28, 28)` 모양의 3D로 표현하기도 한다.
# 예를 들어 `(3, 3, 1)` 모양의 3D 텐서를 `(3, 3)` 모양의 텐서로 표현할 수 있다.

# ```python
# >>> tensor331 = np.array([[[1], [2], [3]],
#                           [[4], [5], [6]],
#                           [[7], [8], [9]]])
# >>> tensor331.reshape(3, 3)
# np.array([[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]])
# ```

# **5D 텐서 실전 예제: 동영상 데이터 활용**
# 
# 동영상은 프레임<font size='2'>frame</font>으로 구성된 순차 데이터다.
# 프레임은 한 장의 컬러 이미지이며, 
# `(높이, 너비, 컬러 채널 수)` 모양의 3D 텐서로 표현된다.
# 따라서 하나의 동영상은 `(프레임 수, 높이, 너비, 컬러 채널 수)` 모양의 4D 텐서로
# 표현된다.
# 이제 여러 개의 동영상으로 이루어진 데이터셋은 
# `(동영상 수, 프레임 수, 높이, 너비, 컬러 채널 수)` 모양의 5D 텐서로 표현된다.
# 
# 예를 들어, `144x256` 크기의 프레임으로 구성된 60초 동영상이 초당 4개의 프레임을 사용한다면
# 동영상 한 편은 `(240, 144, 256, 3)` 모양의 4D 텐서로 표현된다.
# 따라서 동영상 10 편으로 구성된 데이터셋은 `(10, 240, 144, 256, 3)` 모양의 5D 텐서로 표현된다.

# ## 2.3 텐서 연산

# 신경망 모델의 훈련은 기본적으로 텐서와 관련된 몇 가지 연산으로 이루어진다. 
# 예를 들어 이전 신경망에 사용된 케라스 레이어를 살펴보자.
# 
# ```python
# keras.layers.Dense(512, activation="relu")
# keras.layers.Dense(10, activation="softmax")
# ```
# 
# 위 두 개의 층이 하는 일은 데이터셋의 변환이며 실제로 이루어지는 연산은 다음과 같다.
# 
# - 1층: `output1 = relu(dot(input1, W1) + b1)`
# - 2층: `output2 = softmax(dot(input2, W2) + b2)`
# 
# 사용된 세부 연산은 다음과 같다. 
# 
# - 점곱(`dot(input, W)`): 입력 텐서와 가중치 텐서의 곱
# - 덧셈(`dot(input, W) + b`): 점곱의 결과 텐서와 벡터 `b`의 합
# - `relu` 함수: `relu(x) = max(x, 0)`
# - `softmax` 함수: 10개 범주 각각에 속할 확률 계산

# <div align="center"><img src="https://s3-ap-northeast-2.amazonaws.com/opentutorials-user-file/module/3653/9363.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.opentutorials.org/module/3653/22060">생활코딩: 한 페이지 머신러닝</a>&gt;</div></p>

# ### 브로드캐스팅(Broadcasting)
# 
# 두 텐서의 모양이 아래와 같을 때 항목별 연산을 위한 브로드캐스팅이 가능하다.
# 
#     (a, b, … n, n + 1, … m) 와 (n, n + 1, … m)
#     
# - 둘째 텐서에 대해 브로드캐스팅 발생
# - `a`부터 `n-1`까지의 축에 대해 축(axis)이 자동으로 추가됨.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/images/broadcasting10.png" style="width:400px;"></div>

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/images/broadcasting11.png" style="width:400px;"></div>

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/pydata/master/notebooks/images/broadcasting12.png" style="width:400px;"></div>

# ### 텐서 연산의 기하학적 의미

# __이동: 벡터 합__
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/translation.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# __회전: 점곱__
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/rotation.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# __스케일링: 점곱__
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/scaling.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# __아핀 변환__
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/affine_transform.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# __아핀 변환과 relu 활성화 함수__
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/dense_transform.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# __히든 레이어의 중간 결과__

# <div align="center"><img src="https://lh4.googleusercontent.com/VqlLh386Wb-Q924H8o7dFLbmLapWkeiDzh2i-m90fhsGyf-Wxk-LZrCRQLNE5xaYh5mR0n_RX--IjREqraKZD908ko8u_utnKas_XVLm3QUSQcvTN5_VlRKDxQdt55P-B7wUYij1jOA" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.opentutorials.org/module/3653/22060">생활코딩: 한 페이지 머신러닝</a>&gt;</div></p>

# ### 딥러닝의 기하학적 의미
# 
# - 신경망은 기본적으로 앞서 언급된 텐서 연산의 조합에 불과함.
# - 고차원 공간에서의 매우 복잡한 기하학적 변환 = 단순한 텐서 연산의 조합
# - 예제: 3차원 매니폴드
#     - 빨간 종이와 파란 종이 두 장을 겹쳐 뭉친 입력값
#     - 연속된 종이 펼치기 과정을 이용하여 명료하게 구분되는 두 장의 종이로 펼치기
#     - 펼치기 과정에 사용된 과정: 손가락으로 조금씩 펼치기. 부분별로 다른 방식의 펼치기 기법 사용.
#     - 딥러닝 모델의 많은 층(layer)에서 이루어지는 일과 크게 다르지 않음.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch02-geometric_interpretation_4.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ## 2.4 신경망의 엔진: 그레이디언트 기반 최적화

# ### 손실함수
# 
# - 모델에 사용되는 모든 가중치(parameters)에 대한 함수
#     - 입력값: 데이터 묶음. 묶음 크기는 `batch_size`로 지정.
#     - 함숫값: 모델의 예측치와 실제 타깃 사이의 오차.
# - 미분가능이어야 함.
# - __주의사항__: 모델 성능 평가(`metrics`)에 사용되는 정확도 등은 가중치에 대한 미분가능 함수가 아님.

# ### 그레이디언트, 경사하강법, 학습(훈련)
# 
# - __그레이디언트__(gradient): 가중치를 조금 변화시켰을 때 손실값이 어떻게 변하는지 설명
# - __백워드 패스__(backward pass): 가중치에 대한 손실함수의 그레이디언트 계산
# - __경사하강법__(gradient descent): 그레이디언트가 주는 정보를 이용하여
#     손실 함숫값이 낮아지도록 모든 가중치를 __동시에 조금씩__ 업데이트. 
# - __역전파__(backpropagation): 모든 가중치를 계산된 그레이디언트의 반대 방향으로 
#     __학습률__(learning rate)에 비례하여  업데이트 하는 과정.
# - __학습__(learning) 또는 __훈련__(training): __임의로 선택__된 지정된 크기의 데이터 묶음을 
#     대상으로 하는 손실 함숫값 계산과 경사하강법 적용을 반복하는 과정.
# - 최종적으로 손실 함숫값이 최저가 되도록 하는 가중치를 사용하는 모델 완성.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch02-gradient_descent_3d.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 배치, 미니 배치, 또는 확률적 경사하강법
# 
# 묶음(배치)의 크기에 따라 아래 세 종류의 경사하강법이 활용된다.
# 
# - 배치 경사하강법: `batch_size`가 전체 훈련 세트의 크기
# - 미니배치 경사하강법: `batch_size`가 몇 십에서 몇 백.
# - 확률적 경사하강법(SGD): `batch_size = 1`

# ### 옵티마이저(optimizer)와 역전파(backpropagation)
# 
# - 옵티마이저
#     - 경사하강법과 역전파를 실행하는 알고리즘
#     - Adagrad, RMSprop 보다 빠르고 효율적으로 작동하는 알고리즘 활용
# 
# - 역전파(backpropagation) 원리
#     - 손실함수의 그레이디언트를 __연쇄 법칙__(chain rule)을 이용하여 계산

# <div align="center"><img src="https://matthewmazur.files.wordpress.com/2015/03/nn-calculation.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Matt Mazur: A Step by Step Backpropagation Example</a>&gt;</div></p>

# ### 텐서플로우의 그레이디언트 테이프
# 
# - 미분 자동화: 텐서플로우 등이 역전파에 필요한 미분을 자동으로 해결해줌. 
# - __그레이디언트 테이프__(gradient tape): 임의의 텐서 연산에 대해 원하는 변수에 대한
#     그레이디언트를 미리 계산해서 기억해두는 독립적인 장치
# - 케라스가 지원하는 옵티마이저는 내부에서 텐서플로우의 그레이디언트 테이프를 활용함.
