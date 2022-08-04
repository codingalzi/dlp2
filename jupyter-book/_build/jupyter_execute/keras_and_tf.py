#!/usr/bin/env python
# coding: utf-8

# (ch:keras-tf)=
# # 케라스와 텐서플로우

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 케라스와 텐서플로우](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-keras_and_tf.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# 케라스와 텐서플로우를 이용한 딥러닝의 활용법을 소개한다.

# ## 텐서플로우 소개

# 텐서플로우는 파이썬 기본 머신러닝 플랫폼이며,
# 머신러닝 모델의 훈련에 필요한 계산에 필요한 텐서 연산을 지원한다.
# 넘파이(Numpy) 패키지와 유사하지만 보다 많은 기능을 제공한다. 
# 
# - 그레이디언트 자동 계산
# - GPU, TPU 등 고성능 병렬 하드웨어 가속기 활용 가능
# - 여러 대의 컴퓨터 또는 클라우드 컴퓨팅 서비스 활용 가능
# - C++(게임), 자바스크립트(웹브라우저), TFLite(모바일 장치) 등 다른 언어가 선호되는 
#     도메인 특화 프로그램에 쉽게 이식 가능
# 
# 텐서플로우는 또한 단순한 패키지 기능을 넘어서는 머신러닝 플랫폼 역할도 수행한다.
# - TF-Agents: 강화학습 연구 지원
# - TFX: 머신러닝 프로젝트 운영 지원
# - TensorFlow-Hub: 사전 훈련된 머신러닝 모델 제공

# ## 케라스 소개

# 딥러닝 모델 구성 및 훈련에 단순하지만 활용성이 높은 다양한 수준의 API를 제공한다.
# 원래 텐서플로우와 독립적으로 개발되었지만 텐서플로우 2.0부터 텐서플로우 라이브러리의 최상위 프레임워크(framework)로 포함됐다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/keras_and_tf.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ## 딥러닝 주요 라이브러리 약력

# - 2007년: 씨아노(Theano) 공개. 텐서를 이용한 계산 그래프, 미분 자동화 등을 최초로 지원한 딥러닝 라이브러리.
# - 2015년 3월: 케라스 라이브러리 공개. Theano를 백앤드로 사용하는 고수준 패키지.
# - 2015년 11월: 텐서플로우 라이브러리 공개.
# - 2016년: 텐서플로우가 케라스의 기본 백엔드로 지정됨.
# - 2016년 9월: 페이스북이 개발한 PyTorch 공개.
# - 2017년: Theano, 텐서플로우, CNTK(마이크로소프트), MXNet(아마존)이 케라스의 백엔드로 지원됨.
#     현재 Theano, CNTK 등은 더 이상 개발되지 않으며, MXNet은 아마존에서만 주로 사용됨.
# - 2018년 3월: PyTorch와 Caffe2를 합친 PyTorch 출시(페이스북과 마이크로소프트의 협업)
# - 2019년 9월: 텐서플로우 2.0부터 케라스가 텐서플로우의 최상위 프레임워크로 지정됨.

# :::{admonition} 텐서플로우 대 파이토치
# :class: info
# 
# 파이토치<font size='2'>PyTorch</font> 또한 텐서 연산을 지원하는 딥러닝 라이브러리다.
# 텐서플로우와 케라스의 조합이 강력하지만 신경망의 보다 섬세한 조정은 약하다는 지적을 많이 받는 반면에
# 파이토치는 상대적으로 보다 자유롭게 신경망을 구성할 수 있다는 장점이 많이 언급된다.
# 텐서플로우와 케라스의 조합이 여전히 보다 많이 사용되지만 파이토치의 비중 또한 점점 늘고 있다.
# :::

# ## 딥러닝 작업환경

# 딥러닝 신경망 모델의 훈련을 위해서 GPU를 활용할 것을 강력히 추천한다.
# GPU를 사용하지 않으면 모델의 훈련이 너무 느려 제대로 활용할 수 없을 것이다.
# [구글 코랩](https://colab.research.google.com/?hl=ko)을 이용하면
# 특별한 준비가 필요 없이 바로 신경망 모델을 GPU와 함께 훈련시킬 수 있다.
# 구글 코랩은 주피터 노트북을 사용하는데, 주피터 노트북 사용법과
# 구글 코랩에서 GPU를 이용하는 방법을 여기서는 설명하지 않는다.
# 필요한 경우 인터넷에서 쉽게 관련 내용을 찾아볼 수 있다.
# 
# 하지만 딥러닝 모델 훈련을 많이 시키려면 NVIDIA 그래픽카드가 장착된 
# 개인용 컴퓨터를 활용하는 것이 좋다.
# 운영체제는 [Ubuntu](https://ubuntu.com/download/desktop) 또는 [WSL 2(Windows Subsystem for Linux 2)](https://docs.microsoft.com/ko-kr/windows/wsl/install) 사용할 것을 추천한다.
# 
# 윈도우 10/11에서 GPU를 지원하는 텐서플로우를 설치하는 가장 간단한 방식은
# [conda를 활용한 gpu 지원 tensorflow 설치 요령](https://github.com/ageron/handson-ml3/issues/21#issuecomment-1177864010)과
# [Anaconda와 conda 환경 활용](https://github.com/ageron/handson-ml3/blob/main/INSTALL.md)을 참고한다.
# 
# 보다 전문적인 딥러닝 연구를 위해 대용량의 메모리와 고성능의 CPU, GPU가 필요한 경우
# 비용이 들기는 하지만
# [구글 클라우드 플랫폼](https://cloud.google.com/) 또는 
# [아마존 웹서비스(AWS EC2)](https://aws.amazon.com/ko/?nc2=h_lg)를
# 단기간동안 고성능 컴퓨터를 활용할 수 있다.

# ## 순수 텐서플로우 사용법 기초

# 케라스를 전혀 이용하지 않으면서 신경망 모델을 지정하고 훈련시킬 수 있다.
# 하지만 다음 개념, 기능, 도구를 모두 직접 구현해야 한다.
# 
# - 가중치, 편향 등을 저장할 텐서 지정
# - 덧셈, 행렬 곱, `relu()` 함수 등 정의
# - 역전파 실행
# - 층과 모델
# - 손실 함수
# - 옵티마이저
# - 평가지표
# - 훈련 루프

# **상수 텐서와 변수 텐서**
# 
# 텐서플로우 자체로 두 종류의 텐서 자료형을 지원한다.
# 사용법은 기본적으로 넘파이 어레이와 유사하지만 GPU 연산과 그레이디언트 자동계산 등
# 신경망 모델 훈련에 최적화된 기능을 제공한다.
# 
# - `tf.Tensor` 자료형
#     - 상수 텐서
#     - 입출력 데이터 등 변하지 않는 텐서로 사용. 
#     - 불변 자료형
# - `tf.Variable` 자료형
#     - 변수 텐서
#     - 모델의 가중치, 편향 등 업데이트가 되는 텐서로 사용. 
#     - 가변 자료형

# **텐서 연산**
# 
# 덧셈, relu, 점곱 등 텐서 연산은 기본적으로 넘파이 어레이 연산과 동일하다.

# **`GradientTape` 활용**

# 넘파이 어레이와의 가장 큰 차이점은 
# 그레이디언트 테이프 기능을 이용하여 변수 텐서에 의존하는 미분가능한 
# 함수의 그레이디언트 자동 계산이다. 
# 예를 들어 아래 코드는 제곱 함수의 $x = 3$에서의 미분값인 6을 계산한다.
# 
# $$
# f(x) = x^2 \quad \Longrightarrow \quad \nabla f(x) = \frac{df(x)}{dx} = 2x
# $$

# ```python
# >>> input_var = tf.Variable(initial_value=3.)
# >>> with tf.GradientTape() as tape:
# >>>     result = tf.square(input_var)
# >>> gradient = tape.gradient(result, input_var)
# >>> print(gradient)
# tf.Tensor(6.0, shape=(), dtype=float32)
# ```

# 그레이디언트 테이프 기능을 이용하여 신경망 모델 훈련 중에
# 손실 함수의 그레이디언트를 계산한다.
# 
# ```python
# gradient = tape.gradient(loss, weights)
# ```

# :::{admonition} 상수 텐서와 그레이디언트 테이프
# :class: info
# 
# 상수 텐서에 대해 그레이디언트 테이프를 이용하려면 `tape.watch()` 메서드로 감싸야 한다.
# 
# ```python
# input_const = tf.constant(3.)
# with tf.GradientTape() as tape:
#     tape.watch(input_const)
#     result = tf.square(input_const)
# gradient = tape.gradient(result, input_const)
# ```
# 
# 이유는 모델의 가중치와 편향 등 모델 훈련에 중요한 텐서들에 대해 미분 연산을
# 집중하기 위해서이다. 그렇지 않으면 너무 많은 계산을 해야 한다.
# :::

# **순수 텐서플로우로 선형 분류기 구현**

# 케라스를 전혀 사용하지 않으면서 간단한 선형 분류기를 구현하는 과정을 통해
# 텐서플로우 API의 기본 기능을 살펴 본다.

# *데이터셋 생성*

# - `np.random.multivariate_normal()`
#     - 다변량 정규분포를 따르는 데이터 생성
#     - 평균값과 공분산 지정 필요
# - 음성 데이터셋
#     - 샘플 수: 1,000
#     - 평균값: `[0, 3]`
#     - 공분산: `[[1, 0.5],[0.5, 1]]`
# - 양성 데이터셋
#     - 샘플 수: 1,000
#     - 평균값: `[3, 0]`
#     - 공분산: `[[1, 0.5],[0.5, 1]]`

# In[1]:


num_samples_per_class = 1000

# 음성 데이터셋
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)

# 양성 데이터셋
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)


# 두 개의 `(1000, 2)` 모양의 양성, 음성 데이터셋을 하나의 `(2000, 2)` 모양의 데이터셋으로 합치면서
# 동시에 자료형을 `np.float32`로 지정한다. 
# 자료형을 지정하지 않으면 `np.float64`로 지정되어 보다 많은 메모리와 실행시간을 요구한다.

# In[28]:


inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)


# 음성 샘플의 타깃은 0, 양성 샘플의 타깃은 1로 지정한다.

# In[29]:


targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))


# 양성, 음성 샘플을 색깔로 구분하면 다음과 같다.
# 
# - `inputs[:, 0]`: x 좌표
# - `inputs[:, 1]`: x 좌표
# - `c=targets[:, 0]`: 0 또는 1에 따른 색상 지정

# In[30]:


import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()


# *가중치 변수 텐서 생성*

# In[31]:


input_dim = 2     # 입력 샘플의 특성이 2개
output_dim = 1    # 하나의 값으로 출력

# 가중치: 무작위 초기화
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))

# 편향: 0으로 초기화
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# *예측 모델(함수) 선언*
# 
# 아래 함수는 하나의 층만 사용하는 모델이 출력값을 계산하는 과정이다.

# In[32]:


def model(inputs):
    return tf.matmul(inputs, W) + b


# *손실 함수: 평균 제곱 오차(MSE)*
# 
# - `tf.reduce_mean()`: 텐서에 포함된 항목들의 평균값 계산.
#     넘파이의 `np.mean()`과 결과는 동일하지만 텐서플로우의 텐서를 대상으로 함.

# In[33]:


def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


# *훈련 단계*
# 
# 하나의 배치에 대해 예측값을 계산한 후에 손실 함수의 그레이디언트를 이용하여 가중치와 편향을 업데이트한다. 

# In[34]:


learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


# *배치 훈련*
# 
# 배치 훈련을 총 40번 반복한다.

# In[35]:


for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")


# 훈련상태를 보면 여전히 개선의 여지가 보인다. 따라서 학습을 좀 더 시켜본다. 

# In[36]:


for step in range(100):
    loss = training_step(inputs, targets)
    if step % 10 == 0:
        print(f"Loss at step {step}: {loss:.4f}")


# *예측*

# In[37]:


predictions = model(inputs)


# 예측 결과를 확인하면 다음과 같다.
# 예측값이 0.5보다 클 때 양성으로 판정하는 것이 좋은데
# 이유는 샘플들의 레이블이 0 또는 1이기 때문이다.
# 모델은 훈련과정 중에 음성 샘플은 최대한 0에, 
# 양성 샘플은 최대한 1에 가까운 값으로 예측하여 손실값이 줄도록 
# 노력하며, 옵티마이저가 그렇게 유도한다.
# 따라서 0과 1의 중간값인 0.5가 판단 기준으로 적절하다.

# In[38]:


plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()


# 결정 경계를 직선으로 그리려면 아래 식을 이용한다.
# 
# ```python
# y = - W[0] /  W[1] * x + (0.5 - b) / W[1]
# ```
# 
# 이유는 위 모델의 예측값이 다음과 같이 계산되며,
# 
# ```python
# W[0]*x + W[1]*y + b
# ```
# 
# 위 예측값이 0.5보다 큰지 여부에 따라 음성, 양성이 판단되기 때문이다.

# In[39]:


x = np.linspace(-1, 4, 100)
y = - W[0] /  W[1] * x + (0.5 - b) / W[1]

plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)


# ## 3.6 케스의 핵심 API  이해

# ### 신경망 모델 훈련 핵심 2
# 
# 1. 층(layer)과 모델: 층을 적절하게 쌓아 모델 구성
# 1. 손실 함수(loss function): 학습 방향을 유도하는 피드백 역할 수행
# 1. 옵티마이저(optimizer): 학습 방향을 정하는 기능 수행
# 1. 메트릭(metric): 정확도 등 모델 성능 평가 용도
# 1. 훈련 반복(training loop): 미니 배치 경사하강법 실행

# ### 층(layer)의 역할
# 
# - 모델의 상태(지식)로 사용되는 가중치(weight)와 편향(bias) 저장
# - 데이터 표현 변환(forwardd pass)
# - 케라스 활용 딥러닝 모델: 호환 가능한 층들의 적절한 연결

# #### 층의 종류와 처리 가능 텐서
# 
# - `Dense` 클래스를 사용하는 밀집층(dense layer): 
#     `(샘플수, 특성수)` 모양의 2D 텐서로 제공된 데이터셋
# - `LSTM` 클래스, `Conv1D` 클래스 등을 사용하는 순환층(recurrent layer): 
#     `(샘플수, 타임스텝수, 특성수)` 모양의 3D 텐서로 제공된 순차 데이터셋
# - `Cons2D` 클래스 등을 사용하는 층: 
#     `(샘플수, 가로, 세로, 채널수)` 모양의 4D 텐서로 제공된 이미지 데이터셋

# #### `Layer` 클래스와 `__call__()` 메서드
# 
# - 케라스에서 사용되는 모든 층에 대한 부모 클래스
# - `__call__()` 메서드의 역할
#     - 가중치와 편향 벡터 생성 및 초기화
#     - 입력 데이터를 출력 데이터로 변환

# #### `__call__()` 메서드의 대략적 정의
# 
# ```python
# def __call__(self, inputs):
#     if not self.built:
#         self.build(inputs.shape)
#         self.built = True
# return self.call(inputs)
# ```

# - `self.built`: 가중치와 편향 벡터가 초기화가 되어 있는지 여부 기억
# - `self.build(inputs.shape)`: 입력 배치 데이터셋(`inputs`)의 모양(shape) 정보 이용
#     - 가중치 텐서 생성 및 무작위적으로 초기화
#     - 편향 텐서 생성 및 0벡터로 초기화
# - `self.call(inputs)`: 출력값 계산(forward pass)
#     - 아핀 변환 및 활성화 함수 적용

# ### 층에서 모델로
# 
# - 입렵값을 보고 바로 입력값의 모양 확인
# - MNIST 모델 사용된 `Dense` 클래스처럼 입력 데이터에 정보 미리 요구하지 않음

# ```python
# from tensorflow import keras
# from tensorflow.keras import layers
# 
# model = keras.Sequential([
#     layers.SimpleDense(512, activation="relu"),
#     layers.SimpleDense(10, activation="softmax")
# ])
# ```

# #### 딥러닝 모델
# 
# - 층으로 구성된 그래프
# - 예제: `Sequential` 모델
#     - 층을 일렬로 쌓은 신경망 제공
#     - 아래 층에서 전달한 값을 받아 변환한 후 위 층으로 전달
# - 예제: 트랜스포머(Transformer)

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/transformer0001.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# #### 망 구성방식과 가설 공간
# 
# - 모델의 학습과정은 층을 어떻게 구성하였는가에 전적으로 의존함.
# - 여러 개의 `Dense` 층을 이용한 `Sequential` 모델
#     - 아핀 변환,`relu()` 등의 활성화 함수를 연속적으로 적용한 데이터 표현 변환
# - 다른 방식으로 구성된 모델: 다른 방식으로 텐서 표현 변환
# - 이렇듯 층을 구성하는 방식에 따라 텐서들이 가질 수 있는 표현들의 공간이 정해짐.
# - '**망 구성방식(network topology)에 따른 표현 가설 공간(hypothesis space)**'이 지정됨.
# - 신경망의 구성
#     - 주어진 데이터셋과 모델의 목적에 따라 결정됨.
#     - 특별한 규칙 또는 이론은 없음.
#     - 이론 보다는 많은 실습을 통한 경험에 의존

# ### 모델 컴파일
# 
# 모델의 구조를 정의한 후에 아래 세 가지 설정을 추가로 지정해야 함.
# 
# - 옵티마이저(optimizer): 모델의 성능을 향상시키는 방향으로 가중치를 업데이트하는 알고리즘
# - 손실함수(loss function): 훈련 중 모델의 성능 얼마 나쁜가를 측정하는 기준. 
#     미분가능이어야 하며 옵티마이저가 경사하강법을 활용하여 손실함숫값을 줄이는 방향으로 작동함.
# - 평가지표(metrics):: 훈련과 테스트 과정을 모니터링 할 때 사용되는 모델 평가 지표. 
#     옵티마이저 또는 손실함수와 일반적으로 상관 없음.

# ### `fit()` 메서드 작동법
# 
# 모델을 훈련시키려면 `fit()` 메서드를 적절한 인자들과 함께 호출해야 함.
# 
# - 훈련 세트: 보통 넘파이 어레이 또는 텐서플로우의 `Dataset` 객체 사용
# - 에포크(`epochs`): 전체 훈련 세트를 몇 번 훈련할 지 지정
# - 배치 크기(`batch_size`): 배치 경사하강법에 적용될 배치(묶음) 크기 지정
# 
# 아래 코드는 앞서 넘파이 어레이로 생성한 (2000, 2) 모양의 양성, 음성 데이터셋을 대상으로 훈련한다. 

# ### 검증 세트 활용
# 
# 훈련된 모델이 완전히 새로운 데이터에 대해 예측을 잘하는지 여부를 판단하려면
# 전체 데이터셋을 훈련 세트와 **검증 세트**로 구분해야 함.
# 
# - 훈련 세트: 모델 훈련에 사용되는 데이터셋
# - 검증 세트: 훈련된 모델 평가에 사용되는 데이터셋

# ```python
# model.fit(
#     training_inputs,
#     training_targets,
#     epochs=5,
#     batch_size=16,
#     validation_data=(val_inputs, val_targets)
# )
# ```
