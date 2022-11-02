#!/usr/bin/env python
# coding: utf-8

# (ch:computer-vision-intro)=
# # 컴퓨터 비전 기초: 합성곱 신경망

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 컴퓨터 비전 기초: 합성곱 신경망](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-computer_vision_intro.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# - 합성곱 신경망(convnet) 소개
# - 데이터 증식
# - convnet 재활용: 특성 추출과 모델 미세 조정<font size='2'>fine-tuning</font>
# 

# ## 합성곱 신경망 소개

# 2011년부터 2015년 사이에 벌어진 
# 컴퓨터 비전 분야에서의 딥러닝의 획기적 활용이 딥러닝 분야의 약진을 불러왔다.
# 현재 사진 앱, 사진 검색, 유튜브, 카메라 필터, 글자 인식, 자율주행, 로봇공학, 의학 진단 프로그램,
# 얼굴 확인, 스마트 팜 등 일상의 많은 영역에서 딥러닝 모델이 사용되고 있다.
# 
# 컴퓨터 비전 분야에서 일반적으로 가장 많이 사용되는 딥러닝 모델은
# **convnet** 또는 **CNN**으로 불리는 
# **합성곱 신경망**<font size='2'>convolutional neural networks</font>이다.
# 여기서는 작은 크기의 훈련 데이터셋을 이용하여 이미지 분류 문제에
# convnet을 적용하는 방법을 소개한다.

# **예제: MNIST 데이터셋 분류 convnet 모델**
# 
# - `Input()`의 `shape`: `(28, 28, 1)`
# - `Conv2D`와 `MaxPooling2D` 층을 함수형 API 방식으로 층쌓기
#     - 채널수: `filters` 인자로 결정
#     - `kernel_size=3`에 유의할 것.
#     - 출력값: 3D 텐서 (높이, 너비, 채널수). `filers`와 `kernel_size`에 의존.

# ```python
# # 입력층
# inputs = keras.Input(shape=(28, 28, 1))
# 
# # 은닉층
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# 
# # 출력층으로 넘기기 전에 1차원 텐서로 변환
# x = layers.Flatten()(x)
# 
# # 출력층
# outputs = layers.Dense(10, activation="softmax")(x)
# 
# # 모델
# model = keras.Model(inputs=inputs, outputs=outputs)
# ```

# **모델 구성 요약**

# ```python
# >>> model.summary()
# Model: "model" 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param # 
# ================================================================= 
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0 
# _________________________________________________________________
# conv2d (Conv2D)              (None, 26, 26, 32)        320 
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0 
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496 
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0 
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856 
# _________________________________________________________________
# flatten (Flatten)            (None, 1152)              0 
# _________________________________________________________________
# dense (Dense)                (None, 10)                11530 
# =================================================================
# Total params: 104,202 
# Trainable params: 104,202 
# Non-trainable params: 0 
# ```

# **MNIST 이미지 분류 훈련**
# 
# 모델 훈련은 이전과 동일하다.

# ```python
# model.compile(optimizer="rmsprop",
#     loss="sparse_categorical_crossentropy",  # 레이블이 정수인 경우
#     metrics=["accuracy"])
# 
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
# ```

# **훈련된 convnet 평가**
# 
# 테스트셋에 대한 성능이 이전에 사용한 `Sequential` 모델보다 훨씬 좋다.

# ### 합성곱 연산

# - `Dense` 층: 입력값 전체를 학습 대상으로 삼는다. 
#     예를 들어, MNIST의 경우 숫자 이미지 전체를 대상으로 학습한다.
# - `Conv2D` 층: 예를 들어 `kernel_size=3`으로 설정된 경우
#     `3x3` 크기의 국소적 특성들을 대상으로 학습한다.

# **Conv2D 모델의 장점**

# `Conv2D` 층의 장점은 크게 다음 두 가지로 요약된다.
# 
# 첫째, 패턴의 위치와 무관하다.
# 한 번 인식된 패턴은 다른 위치에서도 인식된다.
# 따라서 적은 수의 샘플을 이용하여 일반화 성능이 높은 모델을 훈련시킬 수 있다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/local_patterns.jpg" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 둘째, 패턴 공간의 계층을 파악한다.
# 위 층으로 진행할 수록 보다 복잡한 패턴을 파악한다.
# 이를 **패턴 공간의 계층**(spatial hierarchy of patterns)이라 한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/visual_hierarchy_hires.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **특성맵, 채널, 필터**

# 합성곱 연산의 작동법을 이해하려면 아래 세 개념을 이해해야 한다.
# 
# - **특성맵**<font size='2'>feature map</font>, **채널**<font size='2'>channel</font>:
#     - `높이 x 너비` 모양의 2D 텐서.
#     - 예제: MNIST 데이터셋에 포함된 흑백 이미지 샘플의 경우 `(28, 28)` 모양의 채널 한 개로 구성됨.
#     - 예제: 컬러사진의 경우 세 개의 채널(특성맵)으로 구성됨.
# - **필터**<font size='2'>filter</font>: `kernel_size`를 이용한 3D 텐서. 
#     - 예제: `kernel_size=3`인 경우 필터는 `(3, 3, 입력샘플의깊이)` 모양의 3D 텐서.
#     - 필터 수: `filters` 인자에 의해 결정됨.
# - **출력맵**<font size='2'>response map</font>: 
#     입력 샘플을 대상으로 하나의 필터를 적용해서 생성된 하나의 특성맵.

# :::{admonition} 컬러 이미지와 채널
# :class: info
# 
# 컬러 이미지는 R(red), G(green), B(blue) 세 개의 채털로 구성된다.
# 각각의 채널은 2차원 어레이로 다뤄지기에 하나의 컬러 이미지는 세 개의 채널을 모은 3차원 어레이로 표현된다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-reign_pic_breakdown.png" style="width:700px;"></div>
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-three_d_array.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://e2eml.school/convert_rgb_to_grayscale.html">How to Convert an RGB Image to Grayscale</a>&gt;</div></p>
# 
# 
# :::

# 아래 그림은 입력 샘플에 필터를 적용하기 위해 필터 모양과 동일한 크기의 텐서를 대상으로 필터를 적용하여 **하나의 값**을
# 생성하는 과정을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-filter-product-1.jpg" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.oreilly.com/library/view/fundamentals-of-deep/9781492082170/">Fundamentals of Deep Learning(2판)</a>&gt;</div></p>

# 하나의 필터를 슬라이딩 시키면서 입력 특성맵 전체를 대상으로 위 과정을 적용하여 한 개의 출력맵을
# 생성한다. 
# 아래 그림은 한 개의 채널로 구성된 입력값을 대상으로 하나의 필터를 적용하여 출력맵을 생성하는 과정을 보여준다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-convSobel.gif" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://mlnotebook.github.io/post/CNN1/">Machine Learning Notebook: CNN - Basics</a>&gt;</div></p>

# 예를 들어 6개의 필터를 적용하면 최종적으로 6개의 채널로 구성된 출력 특성맵이 생성된다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-filter-product-2.jpg" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.oreilly.com/library/view/fundamentals-of-deep/9781492082170/">Fundamentals of Deep Learning(2판)</a>&gt;</div></p>

# `Conv2D` 층을 통과할 때 마다 동일한 작업이 반복된다.
# 각 층마다 사용되는 필터의 크기와 수가 다를 뿐이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-cnn-layers.jpg" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.hanbit.co.kr/store/books/look.php?p_code=B7033438574">핸즈온 머신러닝(2판)</a>&gt;</div></p>

# **합성곱 신경망 모델이란?**

# 합성곱 신경망 모델은 바로 이 필터들을 학습시키는 모델을 가리킨다.

# **패딩과 보폭**

# 필터를 적용하여 생성된 출력 특성맵의 모양이 입력 특성맵의 모양과 다를 수 있다.
# 출력 특성맵의 높이와 너비는
# **패딩**<font size='2'>padding</font>의 사용 여부와 
# **보폭**<font size='2'>stride</font>의 크기에 의에 결정된다.
# 
# 다음 세 개의 그림은 입력 특성맵의 높이와 너비가 `5x5`일 때 
# 패딩의 사용 여부와 보폭의 크기에 따라 
# 출력 특성맵의 높이와 너비가 어떻게 달라지는가를 보여준다.

# - 경우 1: 패딩 없음, 보폭은 1.
#     - 출력 특성맵의 깊이와 너비: `3x3`
#     - 즉, 출력 특성맥의 깊이와 너비가 줄어듦.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-padding-stride-01.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.analyticsvidhya.com/blog/2022/03/basics-of-cnn-in-deep-learning/">Basics of CNN in Deep Learning</a>&gt;</div></p>

# - 경우 2: 패딩 없음, 보폭은 2.
#     - 출력 특성맵의 깊이와 너비: `2x2`
#     - 즉, 출력 특성맵의 깊이와 너비가 보폭의 반비례해서 줄어듦.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-padding-stride-02.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.analyticsvidhya.com/blog/2022/03/basics-of-cnn-in-deep-learning/">Basics of CNN in Deep Learning</a>&gt;</div></p>

# - 경우 3: 패딩 있음, 보폭은 1.
#     - 출력 특성맵의 깊이와 너비: `5x5`
#     - 즉, 출력 특성맵의 깊이와 너비가 동일하게 유지됨.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch08-cnn-padding.png" style="width:900px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.researchgate.net/figure/Figure-B2-A-convolutional-filter-with-padding-stride-one-and-filter-size-of-3x3-Image_fig30_324783775">Text to Image Synthesis Using Generative Adversarial Networks</a>&gt;</div></p>

# - 경우 4: 패딩을 사용하고 보폭이 1보다 큰 경우는 굳이 사용할 필요 없음. 
#     이유는 보폭이 1보다 크기에 출력 특성맵의 깊이와 너비가 어차피 보폭에 반비례해서 줄어들기 때문임.

# ### 맥스 풀링 연산

# 합성곱 신경망의 전형적인 모습은 다음과 같이 풀링 층을 합성곱 층 이후에 바로 위치시킨다.

# <div align="center"><img src="http://formal.hknu.ac.kr/handson-ml2/slides/images/ch14/homl14-03.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.hanbit.co.kr/store/books/look.php?p_code=B7033438574">핸즈온 머신러닝(2판)</a>&gt;</div></p>

# 풀링 층 중에서 **맥스 풀링**(max-pooling) 층이 많이 사용된다.
# 맥스 풀링 층은 일정 크기의 영역에서 최댓값만을 선택하여 특성맵의 높이와 너비를 일정 비율로 줄인다. 
# 예를 들어, 아래 맥스 풀링 층은 `2x2`영역에서 최댓값 하나만을 남기고 나머지는 버리며,
# 이 연산을 보폭 2만큼씩 이동하며 입력 특성맵의 모든 영역(`높이x너비`)에 대해 실행한다.
# 맥스 풀링 연산은 입력 특성맵의 채녈 단위로 이루어지기에 채널 수는 변하지 않는다.
# 예를 들어, 만약 `x`가 `(26, 26, 32)` 모양의 3D 텐서이면
# 다음 맥스 풀링 층의 출력값은 `(13, 13, 32)` 모양의 3D 텐서가 된다.
# 
# ```python
# layers.MaxPooling2D(pool_size=2)(x)
# ```

# <div align="center"><img src="http://formal.hknu.ac.kr/handson-ml2/slides/images/ch14/homl14-10.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.hanbit.co.kr/store/books/look.php?p_code=B7033438574">핸즈온 머신러닝(2판)</a>&gt;</div></p>

# 맥스 풀링 층을 합성곱 층(`Conv2D`)과 함께 사용하는 이유는 두 가지이다. 
# 
# - 학습해야할 가중치 파라미터의 수 줄이기.
# - 상위 층으로 갈 수록 입력 특성맵의 보다 넓은 영역에 대한 정보를 얻기 위해.

# 아래 코드는 맥스 풀링 층을 사용하지 않는 경우 가중치 파라미터의 수가 엄청나게 증가함을 잘 보여준다. 
# 
# - 맥스 풀링 사용하는 경우: 104,202개
# - 그렇지 않은 경우: 712,202개

# ```python
# inputs = keras.Input(shape=(28, 28, 1))
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(10, activation="softmax")(x)
# model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs)
# ```

# ```python
# >>> model_no_max_pool.summary()
# Model: "model_1" 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param # 
# =================================================================
# input_2 (InputLayer)         [(None, 28, 28, 1)]       0 
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 26, 26, 32)        320 
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 24, 24, 64)        18496 
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 22, 22, 128)       73856 
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 61952)             0 
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                619530 
# =================================================================
# Total params: 712,202 
# Trainable params: 712,202 
# Non-trainable params: 0 
# ```

# **적절한 `kernel_size`, `stride`, `pool_size` 지정하기**

# - `Conv2D` 층의 기본값: `kernel_size=3`, `stride=1`
# - `MaxPooling2D` 층의 기본값: `pool_size=2`, `strides=2`
#     `strides`의 기본값을 지정하지 않으면 `pool_size`의 값과 동일하게 지정됨.
# 
# 다른 설정 또는 최댓값 대신에 평균값을 사용하는 `AveragePooling2D`를 
# 활용할 수 있으나 케라스의 기본 설정이 일반적으로 가장 좋은 성능을 보인다.

# ## 합성곱 신경망 실전 활용 예제

# ### 작은 데이터셋과 딥러닝 모델

# 이미지 분류 모델을 훈련시킬 때 데이터셋의 크기가 그다지 크지 않은 경우가 일반적이다.
# 즉, 데이터셋의 크기가 적게는 몇 백 개에서 많게는 몇 만 개 정도이다.
# 여기서 훈련시켜야 하는 모델은 개와 고양이 사진을 대상으로 하는 이진분류 합성곱 신경망 모델이다.
# 실전 상황에 맞추기 위해 5천 개의 이미지로 이루어진 작은 데이터셋을 사용한다.
# 
# 합성곱 신경망 모델은 작은 크기의 데이터셋으로도 어느 정도의 성능을 얻을 수 있으며,
# 데이터 증식 기법을 적용하거나 기존에 잘 훈련된 모델을 재활용하여 보다 또는 훨씬 높은 성능의 모델을 구현할 수 있음을 보인다.
# 데이터 증식 기법은 훈련 데이터셋을 크기를 늘리는 기법이며, 
# 사전에 잘 훈련된 모델을 재활용하기 위해 특성 추출 기법과 모델 미세조정 기법을 적용한다.

# ### 데이터 다운로드

# 데이터 과학과 머신러닝과 관련된 다양한 데이터셋과 모델을 활용할 수 있는 
# [캐글<font size='2'>Kaggle</font>](https://www.kaggle.com)에서
# 훈련에 필요한 데이터셋을 다운로드하려면 다음 사항을 먼저 확인해야 한다.
# 
# - 캐글 계정을 갖고 있어야 하며, 로그인된 상태에서 아래 두 과정을 먼저 해결해야 한다.
# - 캐글에 로그인한 후 "Account" 페이지의 계정 설정 창에 있는 "API" 항목에서
#     "Create New API Token"을 생성하여 다운로드한다.
# - [캐글: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/rules)를
#     방문해서 "I Understand and Accept" 버튼을 클릭해야 한다.

# 다운로드된 데이터셋은 총 25,000장의 강아지와 고양이 사진으로 구성되었으며 570MB 정도로 꽤 크다.
# 강아지 사진 고양이 사진이 각각 12,500 장씩 포함되어 있으며, 사진들의 크기가 다음과 같이 일정하지 않다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/dog_and_cat_samples.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **훈련셋, 검증셋, 테스트셋 준비**

# 25,000 장의 사진 중에서 총 5,000 장의 사진만 사용해서 합성곱 신경을 훈련시키려 한다.
# 
# - 훈련셋: 강아지와 고양이 각각 1,000 장
# - 검증셋: 강아지와 고양이 각각 500 장
# - 테스트셋: 강아지와 고양이 각각 1,000 장

# ### 모델 구성

# convnet(합성곱 신경망) 모델은 앞서 설명한대로 `Conv2D`와 `MaxPooling2D` 레이어를
# 연속에서 쌓는 방식을 사용한다.
# 다만, 보다 큰 이미지와 보다 복잡한 문제를 해결하기 위해 모델을 보다 크게 만들기 위해
# `Conv2D`와 `MaxPooling2D` 층을 두 번 더 쌓는다.
# 
# 이렇게 하면 모델의 정보 저장 능력을 키우면서 동시에 특성맵의 크기를 더 작게 만들어
# `Flatten` 층에 최종적으로 `7 x 7` 크기의 않은 특성맵이 전달된다.
# 반면에 특성맵의 깊이(필터 개수)는 32에서 256으로 점차 키운다. 
# 이렇게 층을 쌓아 합성곱 신경망을 구성하는 방식이 매우 일반적이다.
# 
# - 입력층: 입력 샘플의 모양을 `(180, 180, 3)`으로 지정. 픽셀 크기는 임의로 지정함.
#     사진의 크기가 제 각각이기에 먼저 지정된 크기의 텐서로 변환을 해주는 전처리 과정이 필요함.
# - 출력층: 이항분류 모델이기에 한 개의 유닛과 시그모이드 활성화 함수 사용.
# - `Rescaling(1./255)` 층: 0에서 255 사이의 값을 0에서 1 사이의 값으로 변환하는 용도로 사용

# ```python
# # 입력층
# inputs = keras.Input(shape=(180, 180, 3))
# 
# # 은닉층
# x = layers.Rescaling(1./255)(inputs)
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# 
# # 출력층
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# ```

# 강아지와 고양이의 비율이 동일하기에 정확도를 평가지표로 사용한다.

# ```python
# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])
# ```

# ### 데이터 전처리와 모델 훈련

# **데이터 전처리**

# 샘플 사진의 크기가 제 각각이기에 모델의 입력값으로 지정된 크기인 `(180, 180, 3)` 모양의 
# 텐서로 변환해야 한다.
# 케라스의 `image_dataset_from_directory()` 함수를 이용하면 변환 뿐만 아니라
# 지정된 크기의 배치로 구성된 훈련셋, 검증셋, 테스트셋을 쉽게 생성할 수 있다.
# 
# 예를 들어, 아래 코드는 `new_base_dir/train` 라는 디렉토리에 크기가 32인 배치들로 구성된
# 훈련셋을 저장한다.
# 각각의 배치는 `(32, 180, 180, 3)` 모양의 텐서로 저장된다.

# 
# ```python
# train_dataset = image_dataset_from_directory(
#     new_base_dir / "train",
#     image_size=(180, 180),
#     batch_size=32)
# ```

# **모델 훈련**

# `ModelCheckpoint` 콜백을 이용하여 검증셋에 대한 손실값(`"val_loss"`)을
# 기준으로 최고 성능의 모델을 저장한다.

# ```python
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="convnet_from_scratch.keras",
#         save_best_only=True,
#         monitor="val_loss")
#     ]
# 
# history = model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=validation_dataset,
#     callbacks=callbacks)
# ```

# 과대 적합이 10번 정도의 에포크 이후에 빠르게 발생한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-09.png" style="width:800px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 훈련된 최고 성능의 모델에 대한 테스트셋에 대한 정확도가 70% 정도의 정확도로 그렇게 높지 않다.
# 과대적합이 매우 빠르게 발생했기 때문인데 이는 훈련셋의 크기가 2,000 정도로 너무 작기 때문이다.

# ### 데이터 증식

# 데이터 증식 기법을 사용하여 훈련셋의 크기를 키우는 효과를 추가하면,
# 과대적합이 보다 늦게 발생하여 학습된 모델의 성능이 올로간다.
# 
# 데이터 증식을 지원하는 층을 이용하면 쉽게 데이터 증식 기법을 적용할 수 있다.
# 아래 코드의 `data_augmentation`는 Sequential 모델을 이용하여 간단하게 구현된 데이터 증식 층을 가리킨다.
# 
# - `RandomFlip()`: 사진을 50%의 확률로 지정된 방향으로 회전. 
# - `RandomRotation()`: 사진을 지정된 범위 안에서 임의로 좌우로 회전
# - `RandomZoom()`: 사진을 지정된 범위 안에서 임의로 확대 및 축소

# ```python
# data_augmentation = keras.Sequential(
#     [layers.RandomFlip("horizontal"),
#      layers.RandomRotation(0.1),
#      layers.RandomZoom(0.2)]
# )
# ```

# 훈련셋의 첫째 이미지를 대상으로 데이터 증식 층을 아홉 번 적용한 결과를 
# 다음고 같다. 실행결과가 다를 수 있음에 주의하라.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-10.png" style="width:800px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 데이터 증식 층을 이용하여 모델 구성을 다시 한다. 
# 과대적합을 최대한 방지하기 위해 출력층 바로 이전에 드롭아웃(Dropout) 층도 추가한다.

# ```python
# inputs = keras.Input(shape=(180, 180, 3))
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# ```

# 과대 적합이 보다 늦게 발생한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-11.png" style="width:800px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 테스트셋에 대한 정확도가 83% 정도로 올라간다.
# `Conv2D`와 `MaxPooling2D` 층을 더 쌓거나 층에 사용된 필터수를 늘리는 방식으로
# 모델의 성능을 90% 정도까지 끌어올릴 수는 있지만 그 이상은 어려울 것이다.

# ## 모델 재활용

# 적은 양의 데이터셋을 대상으로 훈련하는 것보다 대용량의 데이터셋을 이용하여 훈련하면
# 보다 좋은 성능의 모델을 구현할 수 있다.
# 하지만 대용량의 데이터를 구하기는 매우 어렵거나 아예 불가능할 수 있다.
# 하지만 유사한 목적으로 대용량의 훈련 데이터셋을 이용하여 사전에 훈련된 모델을 재활용하면
# 높은 성능의 모델의 얻을 수 있다.
# 
# 여기서는 잘 알려진 모델 VGG16을 재활용하여 강아지와 고양이 사진을 잘 분류하는
# 모델을 구현하는 다음 두 가지 방식을 소개한다.
# 
# - 전이 학습<font size='2'>transfer learning</font>
# - 모델 미세조정<font size='2'>model fine tuning</font>

# **VGG16 모델**

# VGG16 모델은 [ILSVRC 2014](https://www.image-net.org/challenges/LSVRC/2014/) 
# 경진대회에 참여해서 5등 안에 든 모델이다.
# 당시 훈련에 사용된 데이터셋은 120만 장의 이미지와 1,000개의 클래스로 구성되었으며
# 훈련은 여러 주(weeks)에 걸쳐서 진행되었다. 

# <div align="center"><img src="https://www.image-net.org/static_files/figures/ILSVRC2012_val_00042692.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.image-net.org/challenges/LSVRC/2014/">ILSVRC 2014</a>&gt;</div></p>

# VGG16 모델 구성은 `Conv2D`와 `MaxPooling2D`의 조합으로 이루어졌다(아래 그림 참조).

# <div align="center"><img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://neurohive.io/en/popular-networks/vgg16/">https://neurohive.io/en/popular-networks/vgg16/</a>&gt;</div></p>

# **유명 합성곱 신경망 모델**

# `ketas.applications` 에 포함된 유명 합성곱 신경모델은 다음과 같다.
# 
# - VGG16
# - Xception
# - ResNet
# - MobileNet
# - EfficientNet
# - DenseNet
# - 등등

# **이미지넷(ImagNet) 소개**

# [이미지넷(Imagenet)](https://www.image-net.org/index.php)은
# 대용량의 이미지 데이터셋이며, 
# [ILSVRC](https://www.image-net.org/challenges/LSVRC/index.php) 
# 이미지 분류 경진대회에 사용된다.
# 이미지넷의 전체 데이터셋은 총 2만2천 개 정도의 클래스로 구분되는 동물, 사물 등의 객체를 담은
# 고화질 사진 1500만장 정도로 구성된다.
# 2017년까지 진행된 ILSVRC 경진대회는 보통 1000 개의 클래스로 구분되는 
# 사물을 담은 1백만장 정도 크기의 데이터셋을 이용한다.

# <div align="center"><img src="https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_1k.jpg" style="width:100%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://cs.stanford.edu/people/karpathy/cnnembed/">https://cs.stanford.edu/people/karpathy/cnnembed/</a>&gt;</div></p>

# ### 전이 학습

# **전이 학습 기본 아이디어**

# 사전에 잘 훈련된 모델은 새롭게 구현하고자 하는 모델과 일반적으로 다른 목적으로 구현되었다.
# 하지만 예를 들어 강아지와 고양이를 포함한 동물 및 기타 여러 사물을 대상으로 
# 다중클래스 분류를 목적으로 훈련된 모델은 기본적으로 강아지와 고양이를 
# 분류하는 능력을 갖고 있어야 한다.
# 
# 반면에 이항 분류 모델과 다중클래스 분류 모델은 기본적으로 출력층에서 서로 다른 
# 종류의 값을 출력한다.
# 고양이와 강아지를 포함해서 총 1000개의 사물 클래스로 이미지를 분류하는 모델의 출력층은 
# 1000개의 유닛과 softmax 등과 같은 활성화 함수를 사용할 것이지만
# 고양이-강아지 분류 모델은 1개의 유닛과 sigmoid 등과 같은 활성화 함수를 사용해야 한다.
# 
# 따라서 기존 모델의 출력층을 포함하여 분류값을 직접적으로 예측하는 마지막 몇 개의 층
# (일반적으로 밀집층)을 제외시킨 나머지 합성곱 층으로 이루어진 기저(베이스, base)만을 
# 가져와서 그 위에 원하는 목적에 맞는 층을 새롭게 구성한다(아래 그림 참조).
# 
# 학습 관점에 보았을 때 `Conv2D` 합성곱층과 `MaxPooling2D` 맥스풀링층으로 구성된 기저는
# 이미지의 일반적인 특성을 파악한 정보(가중치)를 포함하고 있기에 
# 강아지/고양이 분류 모델의 기저로 사용될 수 있는 것이다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-12.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **VGG16 모델을 이용한 전이 학습 예제**

# VGG16 합성곱 모델에서 밀집층(dense 층)을 제외한 나머지 합성곱 층으로만 이루어진 모델을 가져온다.

# ```python
# conv_base = keras.applications.vgg16.VGG16(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(180, 180, 3))
# ```

# 모델(층)을 가져올 때 사용된 옵션의 의미는 다음과 같다.
# 
# - `weights="imagenet"`: Imagenet 데이터셋으로 훈련된 모델의 가중치 가져옴.
# - `include_top=False`: 출력값을 결정하는 밀집 층은 제외함.
# - `input_shape=(180, 180, 3)`: 앞서 준비해 놓은 데이터셋을 활용할 수 있도록 지정함. 사용자가 직접 지정해야 함.
#     지정하지 않으면 임의의 크기의 이미지를 처리할 수 있음. 
#     층 별 출력 텐서의 모양의 변화 과정을 확인하기 위해 특정 모양으로 지정함.

# 가져온 모델을 요약하면 다음과 같다.
# 마지막 맥스풀링 층을 통과한 특성맵의 모양은 `(5, 5, 512)`이다.

# ```python
# >>> conv_base.summary()
# Model: "vgg16" 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param # 
# =================================================================
# input_19 (InputLayer)        [(None, 180, 180, 3)]     0 
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 180, 180, 64)      1792 
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 180, 180, 64)      36928 
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 90, 90, 64)        0 
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 90, 90, 128)       73856 
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 90, 90, 128)       147584 
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 45, 45, 128)       0 
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 45, 45, 256)       295168 
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 45, 45, 256)       590080 
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 45, 45, 256)       590080 
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 22, 22, 256)       0 
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 22, 22, 512)       1180160 
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 22, 22, 512)       2359808 
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 22, 22, 512)       2359808 
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 11, 11, 512)       0 
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 11, 11, 512)       2359808 
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 11, 11, 512)       2359808 
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 11, 11, 512)       2359808 
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 5, 5, 512)         0 
# =================================================================
# Total params: 14,714,688 
# Trainable params: 14,714,688 
# Non-trainable params: 0 
# ```

# **특성 추출**

# **특성 추출**<font size='2'>feature extraction</font>은 전이 학습에 사용되는 모델을 이용하여
# 데이터를 변환하는 과정을 의미한다.
# 여기서는 `conv_base` 기저를 특성 추출을 활용하는 두 가지 방식을 소개한다.

# **1) 단순 특성 추출**

# 아래 `get_features_and_labels()` 함수는 
# `conv_base` 모델의 `predict()` 메서드를 이용하여 
# 준비된 훈련 데이터셋을 변환, 
# 즉 특성 추출을 실행한다.
# 단, 레이블은 그대로 변환시키지 않는다.
# 
# - `keras.applications.vgg16.preprocess_input()` 함수는 텐서플로우와 호환이 되도록 데이터를 전처리한다.

# ```python
# def get_features_and_labels(dataset):
#     all_features = []
#     all_labels = []
#     
#     # 배치 단위로 VGG16 모델 적용
#     for images, labels in dataset:
#         preprocessed_images = keras.applications.vgg16.preprocess_input(images)
#         features = conv_base.predict(preprocessed_images)
#         all_features.append(features)
#         all_labels.append(labels)
#         
#     # 생성된 배치를 하나의 텐서로 묶어서 반환
#     return np.concatenate(all_features), np.concatenate(all_labels)
# ```

# 훈련셋, 검증셋, 테스트셋을 변환하면 다음과 같다.

# ```python
# train_features, train_labels =  get_features_and_labels(train_dataset)
# val_features, val_labels =  get_features_and_labels(validation_dataset)
# test_features, test_labels =  get_features_and_labels(test_dataset)
# ```

# 예를 들어, 변환된 강아지/고양이 이미지 샘플 2,000개는 이제 각각 `(5, 5, 512)` 모양을 갖는다.

# ```python
# >>> train_features.shape
# (2000, 5, 5, 12)
# ```

# 변환된 데이터셋을 훈련 데이터셋으로 사용하는 
# 간단한 분류 모델을 구성하여 훈련만 하면 된다.
# `Dropout` 층은 과대적합을 예방하기 위해 사용한다.

# ```python
# # 입력층
# inputs = keras.Input(shape=(5, 5, 512))
# 
# # 은닉층
# x = layers.Flatten()(inputs)
# x = layers.Dense(256)(x)
# x = layers.Dropout(0.5)(x)
# 
# # 출력층
# outputs = layers.Dense(1, activation="sigmoid")(x)
# 
# # 모델
# model = keras.Model(inputs, outputs)
# ```

# 검증셋에 대한 정확도가 97% 정도까지 향상되지만 과대적합이 매우 빠르게 발생함을 확인할 수 있다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-13.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **2) 데이터 증식과 특성 추출**

# 데이터 증식 기법을 활용하려면 
# VGG16 합성곱 기저(베이스)를 구성요소로 사용하는 모델을 직접 정의해야 한다. 
# 다만 앞서 설명한 방식과는 달리 가져온 VGG16 기저에 포함된 파라미터가 새로운
# 모델의 훈련 과정동안 함께 훈련되지 않도록 설정해야 함에 주의해야 한다. 
# 이런 설정을 **동결**(freezing)이라 한다.
# 
# - 기저 동결하기: `trainable=False`로 지정.
# - 입력 데이터의 모양도 미리 지정하지 않음에 주의할 것.

# ```python
# conv_base  = keras.applications.vgg16.VGG16(
#     weights="imagenet",
#     include_top=False)
# 
# # 새로운 학습 금지 설정
# conv_base.trainable = False
# ```

# 동결 해제(`trainable=True`)로 설정하는 경우와 그렇지 않은 경우 학습되어야 하는
# 파라미터의 수가 달라짐을 다음처럼 확인할 수 있다.
# 
# ```python
# >>> conv_base.trainable = True
# >>> print("합성곱 기저의 학습을 허용하는 경우 학습 가능한 파라미터 수: ", 
#           len(conv_base.trainable_weights))
#       
# 합성곱 기저의 학습을 허용하는 경우 학습 가능한 파라미터 수: 26
# ```
# 
# 동결 설정(`trainable=False`)인 경우에 학습되는 파라미터 수가 0이 된다. 
# 
# ```python
# >>> conv_base.trainable = True
# >>> print("합성곱 기저의 학습을 금지하는 경우 학습 가능한 파라미터 수: ", 
#           len(conv_base.trainable_weights))
#       
# 합성곱 기저의 학습을 허용하는 경우 학습 가능한 파라미터 수: 0
# ```

# 아래 모델은 데이터 증식을 위한 층과 VGG16 기저를 함께 이용한다.

# ```python
# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.2),
#     ]
# )
# 
# # 모델 구성
# inputs = keras.Input(shape=(180, 180, 3))
# 
# x = data_augmentation(inputs)                     # 데이터 증식
# x = keras.applications.vgg16.preprocess_input(x)  # VGG16용 전처리
# x = conv_base(x)                                  # VGG16 베이스
# x = layers.Flatten()(x)
# x = layers.Dense(256)(x)
# x = layers.Dropout(0.5)(x)
# 
# outputs = layers.Dense(1, activation="sigmoid")(x) # 출력층
# 
# model = keras.Model(inputs, outputs)
# ```

# 이렇게 훈련하면 재활용된 합성곱 기저에 속한 층은 학습하지 않으며
# 두 개의 밀집층에서만 파라미터 학습이 이뤄진다.
# 과대적합이 보다 늦게 이루어지며 성능도 향상되었다.
# 테스트셋에 대한 정확도가 97.7%까지 향상된다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-14.png" style="width:700px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 모델 미세 조정

# 모델 **미세 조정**(파인 튜닝, fine-tuning)은 특성 추출 방식과는 달리
# 기존 합성곱 모델의 최상위 합성곱 층 몇 개를 동결 해제해서
# 새로운 모델에 맞추어 학습되도록 하는 모델 훈련기법이다.
# 
# 여기서는 아래 그림에처럼 노락색 배경을 가진 상자 안에 포함된 합성곱 층을 
# 동결 해제해서 함께 학습되도록 한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/08-15.png" style="width:200px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 아래 코드는 모든 층에 대해 동결해제를 진행한 후에
# 마지막 4개 층을 제외한 나머지 층에 대해 다시 동결을 설정한다.

# ```python
# conv_base.trainable = True
# for layer in conv_base.layers[:-4]:
#     layer.trainable = False
# ```

# 상위 4개 층만 동결 해제하는 이유는
# 합성곱 신경망의 하위층은 보다 일반적인 형태의 패턴을 학습하는 반면에
# 최상위층은 주어진 문제 해결에 특화된 패턴을 학습하기 때문이다.
# 따라서 이미지넷으로 훈련된 모델 전체를 대상으로 훈련하기 보다는
# 최상위층만 훈련시키는 것이 보다 유용하다.
# 
# 모델 컴파일과 훈련 과정은 이전과 동일하게 진행한다.

# ```python
# model.compile(loss="binary_crossentropy",
#               optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
#               metrics=["accuracy"])
# 
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="fine_tuning.keras",
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=validation_dataset,
#     callbacks=callbacks)
# ```

# 기존 모델을 재활용하여 98%($\pm\!$ 1%)에 육박하는 정확도 성능을 갖는
# 합성곱 신경망 모델을 2,0000개의 이미지만으로 학습시켰음을
# 확인할 수 있다.

# ```python
# model = keras.models.load_model("fine_tuning.keras")
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.3f}")
# ```
