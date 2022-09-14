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

# ## 텐서플로우

# 텐서플로우는 파이썬 기본 머신러닝 플랫폼이며,
# 머신러닝 모델의 훈련에 필요한 텐서 연산을 지원한다.
# 넘파이<font size='2'>Numpy</font> 패키지와 유사하지만 보다 많은 기능을 제공한다. 
# 
# - 그레이디언트 자동 계산
# - GPU, TPU 등 고성능 병렬 하드웨어 가속기 활용 가능
# - 여러 대의 컴퓨터 또는 클라우드 컴퓨팅 서비스 활용 가능
# - C++(게임), 자바스크립트(웹브라우저), TFLite(모바일 장치) 등 다른 언어가 선호되는 
#     도메인 특화 프로그램에 쉽게 이식 가능
# 
# 텐서플로우는 또한 단순한 패키지 기능을 넘어서는 머신러닝 플랫폼 역할도 수행한다.
# 
# - TF-Agents: 강화학습 연구 지원
# - TFX: 머신러닝 프로젝트 운영 지원
# - TensorFlow-Hub: 사전 훈련된 머신러닝 모델 제공

# ## 케라스

# 딥러닝 모델 구성 및 훈련에 단순하지만 활용성이 높은 다양한 수준의 API를 제공하는
# 텐서플로우의 프론트엔드<font size='2'>front end</font> 인터페이스 기능을 수행한다.
# 원래 텐서플로우와 독립적으로 개발되었지만 텐서플로우 2.0부터 텐서플로우 라이브러리의 최상위 프레임워크<font size='2'>framework</font>로 포함됐다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/keras_and_tf.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ## 딥러닝 주요 라이브러리 약력

# - 2007년: 씨아노<font size='2'>Theano</font> 공개. 
#     텐서를 이용한 계산 그래프, 미분 자동화 등을 최초로 지원한 딥러닝 라이브러리.
# - 2015년 3월: 케라스 라이브러리 공개. Theano를 백앤드로 사용하는 고수준 패키지.
# - 2015년 11월: 텐서플로우 라이브러리 공개.
# - 2016년: 텐서플로우가 케라스의 기본 백엔드로 지정됨.
# - 2016년 9월: 페이스북이 개발한 파이토치<font size='2'>PyTorch</font> 공개.
# - 2017년: Theano, 텐서플로우, CNTK(마이크로소프트), MXNet(아마존)이 케라스의 백엔드로 지원됨.
#     현재 Theano, CNTK 등은 더 이상 개발되지 않으며, MXNet은 아마존에서만 주로 사용됨.
# - 2018년 3월: PyTorch와 Caffe2를 합친 PyTorch 출시(페이스북과 마이크로소프트의 협업)
# - 2019년 9월: 텐서플로우 2.0부터 케라스가 텐서플로우의 최상위 프레임워크로 지정됨.

# :::{admonition} 텐서플로우 대 파이토치
# :class: info
# 
# 파이토치 또한 텐서 연산을 지원하는 딥러닝 라이브러리이다.
# 텐서플로우와 케라스의 조합이 강력하지만 신경망의 보다 섬세한 조정은 약하다는 지적을 많이 받는 반면에
# 파이토치는 상대적으로 보다 자유롭게 신경망을 구성할 수 있다고 평가된다.
# 텐서플로우와 케라스의 조합이 여전히 보다 많이 사용되지만 딥러닝 연구에서 파이토치의 활용 또한 점점 늘고 있다.
# :::

# ## 딥러닝 개발환경

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
# 운영체제는 [Ubuntu](https://ubuntu.com/download/desktop) 또는 윈도우 10, 가능하면 윈도우 11을 추천한다.
# 
# - 윈도우 10/11에서 GPU를 지원 텐서플로우 설치: [conda를 활용한 gpu 지원 tensorflow 설치 요령](https://github.com/codingalzi/dlp2) 참고
# 
# - 우분투에서 GPU 지원하는 텐서플로우 설치: [Anaconda와 conda 환경 활용](https://github.com/ageron/handson-ml3/blob/main/INSTALL.md) 참고
# 
# 보다 전문적인 딥러닝 연구를 위해 대용량의 메모리와 고성능의 CPU, GPU가 필요한 경우
# 비용이 들기는 하지만
# [구글 클라우드 플랫폼](https://cloud.google.com/) 또는 
# [아마존 웹서비스(AWS EC2)](https://aws.amazon.com/ko/?nc2=h_lg)를
# 단기간동안 고성능 컴퓨터를 활용할 수 있다.

# ## 순수 텐서플로우 사용법 기초

# 케라스를 전혀 이용하지 않으면서 신경망 모델을 지정하고 훈련시킬 수 있다.
# 하지만 아래에 언급된 개념, 기능, 도구를 모두 직접 구현해야 한다.
# 
# - 가중치, 편향 등을 저장할 텐서 지정
# - 순전파 실행(덧셈, 행렬 곱, `relu()` 함수 등 활용)
# - 역전파 실행
# - 층과 모델
# - 손실 함수
# - 옵티마이저
# - 평가지표
# - 훈련 루프

# ### 상수 텐서와 변수 텐서

# 텐서플로우 패키지가 두 종류의 새로운 텐서 자료형을 지원한다.
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

# ### 텐서 연산

# 덧셈, relu, 점곱 등 텐서 연산은 기본적으로 넘파이 어레이 연산과 동일하다.
# 텐서 연산에 대한 자세한 설명은 
# [텐서 소개(Introduction to Tensors)](https://www.tensorflow.org/guide/tensor) 
# 영어판을 참고하라.

# ### `GradientTape` 활용

# 넘파이 어레이와의 가장 큰 차이점은 
# 그레이디언트 테이프 기능을 이용하여 변수 텐서에 의존하는 미분가능한 
# 함수의 그레이디언트 자동 계산이다. 
# 예를 들어 아래 코드는 제곱 함수의 $x = 3$에서의 미분값인 6을 계산한다.
# 
# $$
# f(x) = x^2 \quad \Longrightarrow \quad \nabla f(x) = \frac{df(x)}{dx} = 2x
# $$

# ```python
# input_var = tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     result = tf.square(input_var)
# gradient = tape.gradient(result, input_var)
# ```

# ```python
# >>> print(gradient)
# tf.Tensor(6.0, shape=(), dtype=float32)
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
# 집중하기 위해서이다. 
# :::

# :::{admonition} 2차 미분
# :class: tip
# 
# 그레이디언트 테이프를 이용하여 
# 2차 미분도 가능하지만 여기서는 관심 대상이 아니다.
# 
# ```python
# time = tf.Variable(0.)
# 
# with tf.GradientTape() as outer_tape:
#     with tf.GradientTape() as inner_tape:
#         position =  4.9 * time ** 2
#     speed = inner_tape.gradient(position, time)
# 
# acceleration = outer_tape.gradient(speed, time)
# ```
# :::

# ### 예제: 순수 텐서플로우 활용 선형 분류기 구현

# 케라스를 전혀 사용하지 않으면서 간단한 선형 분류기를 구현하는 과정을 통해
# 텐서플로우 API의 기본 기능을 살펴 본다.

# :::{admonition} 신경망 모델 훈련 과정
# :class: info
# 
# 신경망 모델의 훈련은 다음 과정을 반복하는 방식으로 진행된다.
# 
# 1. 배치<font size='2'>batch</font> 지정: 훈련 샘플 몇 개로 구성된 텐서 `x`와 해당 샘플들의 타깃값들로 구성된 텐서 `y_true`.
# 1. 순전파<font size="2">forward pass</font>: 예측값 계산, 즉 입력값 `x`에 대한 모델의 예측값 `y_pred` 계산.
#     이때 가중치라는 모델 파라미터가 활용됨.
# 1. 손실값<font size='2'>loss</font> 계산: `y_pred`와 `y_true` 사이의 오차 계산. 모델에 따라 다양한 방식 사용.
# 1. 역전파<font size='2'>backpropagation</font>: 해당 배치에 대한 손실값이 줄어드는 방향으로 모델 파라미터인 가중치를 업데이트.
# 
# 모델의 훈련은 손실값이 최소가 될 때까지 반복된다.
# 손실값을 최소화하는 방향으로 가중치(모델 파라미터)를 업데이트 하기 위해
# 손실함수의 그레이디언트를 활용하여 모든 가중치를 **동시에 조금씩** 업데이트한다.
# 이 과정이 **경사하강법**<font size='2'>gradient descent method</font>이며,
# 백워드 패스와 역전파 단계로 구성된다.
# 
# - **백워드 패스**<font size='2'>backward pass</font>는 
# 가중치에 대한 손실함수의 그레이디언트를 계산하는 과정을 가리키며
# 그레이디언트는 텐서플로우의 `GradientTape` 클래스의 객체에 의해 자동으로 계산되고 관리된다.
# 
# - **역전파**<font size='2'>backpropagation</font>는
# 계산된 그레이디언트와 지정된 학습률<font size='2'>learning rate</font>을 이용하여
# 모든 가중치를 동시에 업데이트 하는 과정이다. 
# 
# **옵티마이저**<font size='2'>optimizer</font>는 경사하강법(백워드 패스, 역전파) 업무를
# 처리하는 알고리즘을 가리키며 momentum optimization, Nesterov Accelerated Gradeitn, 
# AdaGrad, RMSProp, Adam optimization 등 다양한 알고리즘이 존재한다.
# 
# - 경사하강법: 
#     [핸즈온 머신러닝(3판), 4.2절](https://codingalzi.github.io/handson-ml3/training_models.html#sec-gradient-descent)이
#     머신러닝 모델 일반적인 훈련에 사용되는 경사하강법을 쉽게 설명한다.
# - 역전파: 
# [Matt Mazur의 A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)가 신경망 모델의 역전파 과정을 친절히 설명한다.
# - 옵티마이저:
#     [핸즈온 머신러닝(3판)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/), 11장에서 다양한 옵티마이저를 소개한다.
# :::

# **1단계: 데이터셋 생성**

# 아래 사진 모양처럼 양성(노랑색)과 음성(보라색)으로 구분되는 훈련셋을 생성해서
# 훈련셋으로 이용한다.
# 생성되는 훈련셋은 다변량 정규분포를 따르도록 하며,
# 양성과 음성 데이터셋 각각 1,000개의 샘플로 구성된다.
# 데이터셋 생성에 사용되는 공분산은 두 데이터셋에 대해 동일하고 평균값만 서로 다르다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/03-07.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ```python
# num_samples_per_class = 1000
# 
# # 음성 데이터셋
# negative_samples = np.random.multivariate_normal(
#     mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)
# 
# # 양성 데이터셋
# positive_samples = np.random.multivariate_normal(
#     mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=num_samples_per_class)
# ```

# 두 데이터셋을 합쳐서 훈련셋, 즉, 모델의 입력값으로 지정한다.
# 자료형을 `np.float32`로 지정함에 주의하라.
# 그렇게 하지 않으면 `np.float64`로 지정되어 보다 많은 메모리와 실행시간을 요구한다.

# ```python
# inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
# ```

# 음성 샘플의 레이블은 0, 양성 샘플의 레이블은 1로 지정한다.
# 레이블 데이터셋 또한 2차원 어레이로 지정됨에 주의하라.

# ```python
# targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
#                      np.ones((num_samples_per_class, 1), dtype="float32")))
# ```

# **2단계: 선형 회귀 모델 훈련에 필요한 가중치와 편향 변수 텐서 생성**

# 모델 학습에 사용될 가중치와 편향을 변수 텐서로 선언한다.

# ```python
# input_dim = 2     # 입력 샘플의 특성이 2개
# output_dim = 1    # 하나의 값으로 출력
# 
# # 가중치: 무작위 초기화
# W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
# 
# # 편향: 0으로 초기화
# b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
# ```

# **3단계: 모델 선언(포워드 패스 담당)**

# 신경망 모델을 훈련할 때 입력값에 대한 예측값을 계산하는 과정인
# 포워드 패스를 함수로 구현한다.
# 간단한 모델 표현을 위해 활성화 함수는 사용하지 않는다.
# 
# `tf.matmul()` 함수는 넘파이의 점 곱 함수처럼 작동하며 아래 코드에서는
# 행렬의 곱을 나타낸다.

# ```python
# def model(inputs):
#     return tf.matmul(inputs, W) + b
# ```

# **4단계: 손실 함수 지정**

# 타깃과 예측값 사이의 평균 제곱 오차를 손실값으로 사용한다. 
# 아래 코드에서 `tf.reduce_mean()` 함수는 넘파이의 `np.mean()`처럼
# 평균값을 계산하지만 텐서플로우의 텐서를 대상으로 한다.

# ```python
# def square_loss(targets, predictions):
#     per_sample_losses = tf.square(targets - predictions)
#     return tf.reduce_mean(per_sample_losses)
# ```

# **5단계: 훈련 스텝(백워드 패스와 역전파) 지정**

# 하나의 배치에 대해 예측값을 계산한 후에 손실 함수의 그레이디언트를 
# 계산한 후에 가중치와 편향을 업데이트하는 함수를 선언한다.
# 그레이디언트 계산은 그레이디언트 테이프를 이용한다.
# 
# 아래 `training_step()` 함수가 백워드 패스와 역전파를 수행하는 옵티마이저
# 역할을 담당한다.

# ```python
# def training_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs)
#         loss = square_loss(targets, predictions)
#     grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
#     W.assign_sub(grad_loss_wrt_W * learning_rate)
#     b.assign_sub(grad_loss_wrt_b * learning_rate)
#     return loss
# ```

# **6단계: 훈련 루프 지정**

# 반복해서 훈련한 내용을 지정한다.
# 여기서는 설명을 간단하게 하기 위해 미니 배치가 아닌 배치 훈련을 구현한다.
# 전체 훈련셋을 총 40번 반복 학습할 때마다 손실값을 출력하도록 한다.

# ```python
# for step in range(40):
#     loss = training_step(inputs, targets)
#     print(f"Loss at step {step}: {loss:.4f}")
# ```

# **7단계: 결정경계 예측**

# 모델의 예측값이 0.5보다 클 때 양성으로 판정하는 것이 좋은데
# 이유는 샘플들의 레이블이 0 또는 1이기 때문이다.
# 모델은 훈련과정 중에 음성 샘플은 최대한 0에, 
# 양성 샘플은 최대한 1에 가까운 값으로 예측하여 손실값을 최대한 줄여야 하는데
# `training_step()` 함수에서 구현된 경사하강법이 그렇게 유도한다.
# 따라서 예측값이 0과 1의 중간값인 0.5일 때를 결정경계로 사용한다.

# 결정경계를 직선으로 그리려면 아래 식을 이용한다.
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

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/HighResolutionFigures/figure_3-8.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ## 케라스 신경망 모델의 핵심 API

# 신경망 모델은 층<font size='2'>layer</font>으로 구성된다.
# 모델에 사용되는 층의 종류와 층을 쌓는 방식에 따라
# 모델이 처리할 수 있는 데이터와 훈련 방식이 달라진다.
# 모델과 훈련 방식이 정해지면 이후에 훈련을 어떻게 진행할 것인가를 정해야 한다.
# 이또한 많은 선택이 필요하지만 케라스가
# 쉽고 간단하게 활용할 수 있는 다양한 API를 제공한다.

# ### 층

# **층의 기능**

# 층은 입력 데이터를 지정된 방식에 따라 다른 모양의 데이터로 변환하는 포워드 패스 기능을 담당한다.
# 또한 데이터 변환에 사용되는 가중치<font size='2'>weight</font>와 
# 편향<font size='2'>bias</font>을 저장한다.

# **층의 종류**

# 층은 사용되는 클래스에 따라 다양한 형식의 텐서를 취급한다.
# 
# - `Dense` 클래스: 밀집층(dense layer)이며, `(샘플 수, 특성 수)` 모양의 2D 텐서 데이터셋 처리.
# - `LSTM` 또는 `Conv1D` 클래스: 순차 데이터 및 시계열 데이터 분석에 사용되는 순환층이며,
#     `(샘플 수, 타임스텝 수, 특성 수)` 모양의 3D 텐서로 제공된 순차 데이터셋 처리.
# - `Conv2D` 클래스: `(샘플 수, 가로, 세로, 채널 수)` 모양의 4D 텐서로 제공된 이미지 데이터셋 처리.
#     
# 케라스를 활용하여 딥러닝 모델을 구성하는 것은 호환 가능한 층들을 적절하게 연결하여 층을 쌓는 것을 의미한다.

# **`keras.layers.Layer` 클래스**

# 케라스에서 사용되는 모든 층 클래스는 `keras.layers.Layer` 클래스를 상속하며,
# 이를 통해 상속받는 `__call__()` 메서드가 
# 가중치와 편향 텐서를 생성 및 초기화 하고 입력 데이터를 출력 데이터로 변환하는 
# 포워드 패스를 수행한다.
# 단, 가중치와 편향이 이미 생성되어 있다면 새로 생성하지 않고 그대로 사용한다. 
# `keras.layers.Layer` 클래스에서 선언된 `__call__()` 메서드가 하는 일을 간략하게 나타내면 다음과 같다. 
# 
# ```python
# def __call__(self, inputs):
#     if not self.built:
#         self.build(inputs.shape)
#         self.built = True
# return self.call(inputs)
# ```

# 위 코드에 사용된 인스턴스 변수와 메서드는 다음과 같다. 
# 
# - `self.build(inputs_shape)`: 입력 배치 데이터셋의 모양 정보를 이용하여 
#     적절한 모양의 가중치 텐서와 편향 텐서를 생성하고 초기화한다.
#     - 가중치 텐서 초기화: 무작위적(`random_normal`)
#     - 편향 텐서 초기화: 0 벡터(`zeros`)
# - `self.built`: 가중치와 편향이 준비돼 있는지 여부 기억
# - `self.call(inputs)`: 아핀 변환과 활성화 함수를 이용한 포워드 패스,
#     입력 데이터셋을 변환하여, 출력 텐서를 계산한다.

# **`Dense` 클래스 직접 구현하기**

# {numref}`%s절 <sec:nn-mnist>`에서 MNIST 데이터셋을 이용한 분류 모델에 사용된
# 신경망 모델은 `Dense` 클래스 두 개를 연속으로 쌓아 사용한다.

# ```python
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# ```

# `Dense` 클래스와 유사하게 작동하는 클래스를 직접 정의하려면 
# 상속해야 하는 `keras.layers.Layer` 클래스의 `__call()__` 메서드에 의해 호출되는
# `build()` 메서드와 `call()` 메서드를 구현해야 한다.
# 아래 `SimpleDense` 클래스가 `Dense` 클래스의 기능을 단순화하여 구현한다.

# ```python
# class SimpleDense(keras.layers.Layer):
#     def __init__(self, units, activation=None):
#         super().__init__()
#         self.units = units
#         self.activation = activation
# 
#     def build(self, input_shape):
#         input_dim = input_shape[-1]   # 입력 샘플의 특성 수
#         self.W = self.add_weight(shape=(input_dim, self.units),
#                                  initializer="random_normal")
#         self.b = self.add_weight(shape=(self.units,),
#                                  initializer="zeros")
# 
#     def call(self, inputs):
#         y = tf.matmul(inputs, self.W) + self.b
#         if self.activation is not None:
#             y = self.activation(y)
#         return y
# ```

# 두 메서드의 정의에 사용된 매개변수와 메서드는 다음과 같다.
# 
# - `units`: 출력 샘플의 특성 수 지정
# - `activation`: 활성화 함수 지정
# - `input_shape`: 입력값(`inputs`)으로 얻은 입력 배치의 2D 모양 정보. 둘째 항목이 입력 샘플의 특성 수.
# - `add_weight(모양, 초기화방법)`: 지정된 모양의 텐서 생성 및 초기화. `Layer` 클래스에서 상속.

# :::{prf:example} `SimpleDense` 층 활용법
# :label: simpledense
# 
# 아래 코드는 `SimpleDense` 층을 하나 생성한다.
# 층은 입렵값을 처리할 때 입력값의 모양을 확인하기 때문에 
# 2장에서 살펴본 MNIST 모델 사용된 `Dense` 클래스처럼 입력 데이터에 대한 정보를
# 미리 요구하지 않는다.
# 
# ```python
# >>> my_dense = SimpleDense(units=32, activation=tf.nn.relu)
# ```
# 
# 784 개의 특성을 갖는 1,000 개의 샘플로 구성된 데이터셋을 입력값으로 지정한다.
# 
# ```python
# >>> input_tensor = tf.ones(shape=(2, 784))
# ```
# 
# `my_dense`를 함수 호출하듯이 사용하면 출력값이 계산된다.
# 
# ```python
# >>> output_tensor = my_dense(input_tensor)
# ```
# 
# 내부적으로는 `__call__()` 메서드가 호출되어 다음 사항들이 연속적으로 처리된다. 
# 
# - `(784, 32)` 모양의 가중치 텐서 `W` 생성 및 무작위 초기화
# - `(32, )` 모양의 편향 텐서 `b` 생성 및 `0`으로 초기화.
# - 포워드 패스: 생성된 가중치와 편향을 이용하여 출력값 계산
# 
# 층의 출력값은 `(2, 32)` 모양의 텐서다.
# 이유는 784개의 특성이 32개의 특성으로 변환되었기 때문이다.
# 
# ```python
# >>> print(output_tensor.shape)
# (2, 32)
# ```
# :::

# ### 모델

# **층에서 모델로**

# 딥러닝 모델은 층으로 구성된다.
# 앞서 살펴 본 `Sequential` 모델은 층을 일렬로 쌓은 모델이며
# 각각의 층은 아래 층에서 전달한 값을 받아 변환해서 다음 층으로 전달한다.
# 
# 앞으로 층을 구성하는 보다 복잡하고 다양한 방식을 살펴볼 것이다.
# 예를 들어, 아래 예제는 텍스트 번역에 사용되는
# 트랜스포머<font size='2'>Transformer</font> 모델에
# 사용된 층들의 복잡한 구조를 보여준다.

# :::{prf:example} 트랜스포머
# :label: exp-transformer
# 
# {numref}`%s장 자연어 처리 <ch:nlp>`에서 다룰 `TransformerDecoder`의 구조는 다음과 같다.
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/transformer0001.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>
# :::

# **층의 구성과 모델의 학습과정**

# 모델의 학습과정은 층을 어떻게 구성하였는가에 전적으로 의존한다. 
# 앞서 살펴 보았듯이 각각의 층에서 이루어지는 일은 기본적으로 
# **아핀 변환**과 **활성화 함수 적용**이다. 
# 여러 개의 `Dense` 층을 `Sequential` 모델을 이용하여 층을 구성하면 
# 아핀 변환,`relu()` 등의 활성화 함수를 연속적으로 적용하여
# 입력 텐서를 특정 모양의 텐서로 변환한다.
# 
# 반면에 여러 개의 층을 다른 방식으로 구성한 모델은 다른 방식으로 텐서를
# 하나의 표현에서 다른 표현으로 변환한다.
# 예를 들어 {prf:ref}`exp-transformer`의 `TransformerDecoder`는
# 텍스트 번역에 특화된 층의 구성방식을 보여준다.
# 
# 일반적으로 딥러닝 신경망의 구성은 주어진 데이터셋과 모델의 목적에 따라 결정되며
# 특별히 따라야 하는 정해진 규칙은 없다.
# 따라서 모델의 구조는 이론 보다는 많은 실습을 통한 경험에 의존한다.
# 앞으로 많은 예제를 통해 다양한 모델을 구성하는 방식을 배울 것이다.

# **모델 컴파일**

# 모델의 훈련을 위해서 먼저 다음 세 가지 설정을 추가로 지정해야 한다.
# 
# - 손실 함수: 훈련 중 모델의 성능이 얼마나 나쁜지 측정.
#     미분가능한 함수이어야 하며 옵티마이저가 역전파를 통해
#     모델의 성능을 향상시키는 방향으로 모델의 가중치를 업데이트할 때 
#     참고하는 함수임.
# - 옵티마이저: 백워드 패스와 역전파를 담당하는 알고리즘
# - 평가지표: 훈련과 테스트 과정을 모니터링 할 때 사용되는 모델 평가 지표.
#     옵티마이저와 손실함수와는 달리 훈련에 관여하지 않으면서
#     모델 성능 평가에 사용됨.

# 아래 코드는 옵티마이저, 손실 함수, 평가지표를 문자열로 지정한다.
# 
# ```python
# model = keras.Sequential([keras.layers.Dense(1)])
# model.compile(optimizer="rmsprop",
#               loss="mean_squared_error",
#               metrics=["accuracy"])
# ```
# 
# 각각의 문자열은 특정 파이썬 객체를 가리킨다.
# 
# | 문자열 | 파이썬 객체 |
# | :--- | :--- |
# | `"rmsprop"` | `keras.optimizers.RMSprop()` |
# | `"mean_squared_error"` | `keras.losses.MeanSquaredError()` |
# | `"accuracy"` | `keras.metrics.BinaryAccuracy()]` |
# 
# 따라서 지정된 문자열을 사용하는 대신 파이썬 객체를 직접 작성해도 된다.
# 
# ```python
# model.compile(optimizer=keras.optimizers.RMSprop(),
#               loss=keras.losses.MeanSquaredError(),
#               metrics=[keras.metrics.BinaryAccuracy()])
# ```
# 
# 옵티마이저 설정에 기본값과 다른 학습률(`learning_rate`)을 지정하는 경우 또는
# 사용자가 직접 정의한 객체를 사용하려는 경우엔 문자열 대신 직접 파이썬 객체를 지정하는 
# 방식을 사용해야 한다.
# 
# ```python
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
#               loss=사용자정의손실함수객체,
#               metrics=[사용자정의평가지표_1, 사용자정의평가지표_2])
# ```

# 일반적으로 가장 많이 사용되는 옵티마이저, 손실함수, 평가지표는 다음과 같으며
# 앞으로 다양한 예제를 통해 옵티마이저, 손실함수, 평가지표를 적절하게 선택하는 방법을 살펴볼 것이다.
# 
# 옵티마이저:
# 
# - SGD (with or without momentum)
# - RMSprop
# - Adam
# - Adagrad
# 
# 손실 함수:
# 
# - CategoricalCrossentropy
# - SparseCategoricalCrossentropy
# - BinaryCrossentropy
# - MeanSquaredError
# - KLDivergence
# - CosineSimilarity
# 
# 평가지표:
# 
# - CategoricalAccuracy
# - SparseCategoricalAccuracy
# - BinaryAccuracy
# - AUC
# - Precision
# - Recall

# ### 훈련 루프

# 모델을 컴파일한 다음에 `fit()` 메서드를 호출하면
# 모델은 스텝과 에포크 단위로 반복되는 **훈련 루프**<font size='2'>training loop</font>를
# 지정된 횟수만큼 또는 학습이 충분히 이루어졌다는 평가가 내려질 때까지
# 반복하는 훈련을 시작한다.

# **모델 훈련**

# 모델을 훈련시키려면 `fit()` 메서드를 적절한 인자들과 함께 호출해야 한다.
# 
# - 훈련셋과 타깃셋: 보통 넘파이 어레이 또는 텐서플로우의 `Dataset` 객체 사용
# - 에포크(`epochs`): 전체 훈련 세트를 몇 번 훈련할 지 지정
# - 배치 크기(`batch_size`): 배치 경사하강법에 적용될 배치(묶음) 크기 지정
# 
# 아래 코드는 앞서 넘파이 어레이로 생성한 (2000, 2) 모양의 양성, 음성 데이터셋을 대상으로 훈련한다. 
# 
# ```python
# training_history = model.fit(
#     inputs,
#     targets,
#     epochs=5,
#     batch_size=128
# )
# ```

# **훈련 결과**

# 모델의 훈련 결과로 `History` 객체가 반환된다.
# 예를 들어 `History` 객체의 `history` 속성은 에포크별로 계산된 손실값과 평가지표값을
# 사전 자료형으로 가리킨다.

# ```python
# >>> training_history.history
# {'loss': [9.07500171661377,
#   8.722702980041504,
#   8.423994064331055,
#   8.137178421020508,
#   7.8575215339660645],
#  'binary_accuracy': [0.07800000160932541,
#   0.07999999821186066,
#   0.08049999922513962,
#   0.08449999988079071,
#   0.0860000029206276]}
# ```

# **검증 데이터 활용**

# 머신러닝 모델 훈련의 목표는 훈련셋에 대한 높은 성능이 아니라
# 훈련에서 보지 못한 새로운 데이터에 대한 정확한 예측이다.
# 훈련 중에 또는 훈련이 끝난 후에 모델이 새로운 데이터에 대해 정확한 예측을 하는지
# 여부를 판단하도록 할 수 있다.
# 
# 이를 위해 전체 데이터셋을 훈련셋과 검증셋<font size='2'>validation data set</font>으로 구분한다.
# 훈련셋과 검증셋의 비율은 8대2 또는 7대3 정도로 한다.
# 훈련셋이 매우 크다면 검증셋의 비율을 보다 적게 잡을 수 있다.
# 다만 훈련셋과 검증셋이 서로 겹치지 않도록 주의해야 한다.
# 그렇지 않으면 훈련 중에 모델이 검증셋에 포함된 데이터를 학습하기에
# 정확환 모델 평가를 할 수 없게 된다.

# *훈련 중 모델 검증*

# 아래 코드는 미리 지정된 검증셋 `val_inputs`와 검증 타깃값 `val_targets`를
# `validation_data`의 키워드 인자로 지정해서
# 모델 훈련 중에 에포크 단위로 측정하도록 한다.
# 
# ```python
# model.fit(
#     training_inputs,
#     training_targets,
#     epochs=5,
#     batch_size=16,
#     validation_data=(val_inputs, val_targets)
# )
# ```

# *훈련 후 모델 검증*

# 훈련이 끝난 모델의 성능 검증하려면 `evaluate()` 메서드를 이용한다.
# 배치 크기(`batch_size`)를 지정하여 배치 단위로 학습하도록 한다.
# 
# ```python
# >>> loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)
# ```
# 
# 반환값으로 지정된 손실값과 평가지표를 담은 리스트가 생성된다.
# 
# ```python
# >>> print(loss_and_metrics)
# [0.29411643743515015, 0.5333333611488342]
# ```

# ### 예측

# 모델의 훈련과 검증이 완료되면 실전에서 새로운 데이터에 대한 예측을 진행한다.
# 데이터셋에 포함된 모든 데이터에 대한 예측을 한 번에 실행할 수 있으며
# 두 가지 방식이 존재한다.

# **모델 적용**

# 모델을 마치 함수처럼 이용한다. 
# 
# ```python
# predictions = model(new_inputs)
# ```
# 
# 내부적으론 앞서 설명한 `__call()__` 메서드가 실행된다.
# 따라서 `call()` 메서드를 사용하는 포워드 패스가 실행되어
# 예측값이 계산된다.
# 
# 하지만 이 방식은 입력 데이터셋 전체를 대상으로 한 번에 계산하기에
# 데이터셋이 너무 크면 계산이 너무 오래 걸리거나 메모리가 부족해질 수 있다.
# 따라서 배치를 활용하는 `predict()` 메서드를 활용할 것을 추천한다.

# :::{admonition} 모델 함수와 포워드 패스
# :class: tip
# 
# 모델을 함수처럼 이용하는 방식은 포워드 패스와 관련해서 종종 사용된다.
# :::

# **`predict()` 메서드**

# 훈련된 모델의 `predict()` 메서드는 배치 크기를 지정하면
# 배치 단위로 예측값을 계산한다.
# 
# ```python
# predictions = model.predict(new_inputs, batch_size=128)
# ```

# ## 연습 문제

# 1. [텐서 소개(Introduction to Tensors)](https://www.tensorflow.org/guide/tensor)를 학습하라.
# 1. ...
