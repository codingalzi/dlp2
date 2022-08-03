#!/usr/bin/env python
# coding: utf-8

# # 딥러닝 소개

# ## 인공지능, 머신러닝, 딥러닝

# **관계 1: 연구 분야 관점**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml2/master/slides/images/ai-ml-relation.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="http://www.kyobobook.co.kr/readIT/readITColumnView.laf?thmId=00198&sntnId=14142">교보문고(에이지 오브 머신러닝)</a>&gt;</div></p>

# **관계 2: 역사**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml2/master/slides/images/ai-ml-relation2.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/">NVIDIA 블로그</a>&gt;</div></p>

# ### 인공지능

# - 인공지능: 인간의 지적 활동을 모방하여 컴퓨터로 자동화하려는 시도. 머신러닝과 딥러닝을 포괄함.
# - (1950년대) 컴퓨터가 생각할 수 있는가? 라는 질문에서 출발
# - (1956년) 존 맥카시(John McCarthy)
#     - 컴퓨터로 인간의 모든 지적 활동 구현하는 것이 가능하다고 판단.
# - (1980년대까지) __학습__(러닝)이 아닌 모든 가능성을 논리적으로 전개하는 기법 활용
#     - 서양장기(체스) 등에서 우수한 성능 발휘
#     - 반면에 이미지 분류, 음석 인식, 자연어 번역 등 보다 복잡한 문제는 제대로 다루지 못함.

# ### 머신러닝

# 머신러닝은 1990년대에 본격적으로 발전했다. 
# 기본적으로 통계학과 연관되어 출발했지만 다음 두 측면에서 통계학과 다르다. 
# 
# 첫째, 아주 큰 데이터(빅데이터)를 단순히 통계학적으로 다룰 수는 없다.
# 예를 들어, 수 백만장의 사진을 통계 기법으로 다루지 못하지만 머신러닝은 가능하다.
# 
# 둘째, 머신러닝, 특히 딥러닝의 경우 수학적, 통계적 이론 보다는 공학적 접근법이 보다 중요하다.
# 또한 소프트웨어와 하드웨어의 발전이 딥러닝 연구에 중요한 역할을 수행한다.

# **전통적 인공지능 프로그램 대 머신러닝 프로그램**
# 
# - 전통적 인공지능 프로그램
#     - 컴퓨터가 수행해야 할 규칙을 순서대로 적어 놓은 프로그램 작성
#     - 입력값이 지정되면 지정된 규칙을 수행하여 적절한 답을 생성함.
# 
# - 머신러닝 프로그램
#     - 주어진 입력 데이터와 출력 데이터로부터 입력과 출력 사이에 존재하는
#         특정 통계적 구조를 스스로 알아내어
#         이를 이용하여 입력값으로부터 출력값을 생성하는 규칙을 생성함.
#     - 예제: 사진 태그 시스템. 태그 달린 사진 데이터셋을 학습한 후 
#         자동으로 사진의 태그 작성.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-a-new-programming-paradigm.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 학습 규칙과 데이터 표현법

# **머신러닝 모델 학습의 필수 요소**
# 
# - **입력 데이터셋**: 음성 인식 모델을 위한 음성 파일, 이미지 태깅 모델을 위한 사진 등.
# - **타깃 데이터셋**: 음성 인식 작업의 경우 사람이 직접 작성한 글, 
#     이미지 작업의 경우 '강아지', '고양이', 등의 사람이 직접 붙힌 태그.
# - **모델 평가지표**: 출력 예상값과 기대 출력값 사이의 거리(차이) 측정법. 
#     거리를 줄이는 방향으로 알고리즘에 사용되는 파라미터를 반복 수정하는
#     과정을 **학습**이라 부름.

# **데이터 표현법 학습**
# 
# 머신러닝 모델은 입력 데이터를 적절하게 변환하여 원하는 결과를 생성하는
# 규칙을 입력과 타깃 데이터셋을 이용하여 학습해 나간다.
# 즉, 주어진 과제 해결에 가장 적절한 **데이터 표현법**을 모델 학습을 통해 알아낸다.
# 
# 데이터는 다양한 방식으로 표현될 수 있으며
# 과제에 따라 보다 적절한 표현법을 채택해야 한다. 
# 예를 들어, 컬러 사진을 
# 빨간색-초록색-파란색을 사용하는 RGB 방식으로 표현하거나
# 색상-채도-명도를 사용하는 HSV 방식으로 표현할 수 있다.
# 컬러 사진에서 빨간색 픽셀만을 선택하고자 할 때는 RGB 방식으로 컬러 사진을
# 표현하는 것이 좋은 반면에 
# 컬러 사진의 채도를 조정하고자 할 때는 HSV 방식으로 표현된
# 데이터를 사용해야 한다. 

# :::{prf:example} 선형 분류
# :label: exp-lin-reg
# 
# 흑점과 백점이 아래 왼쪽 그림에서첨 분포되어 있다. 
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-learning_representations.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>
# 
# 좌표 $(x, y)$가 주어지면 해당 좌표의 점의 색깔을 예측하는 머신러닝 모델을 구현하고자 한다.
# 이 경우 머신러닝 모델의 학습에 필요한 세 요소가 다음과 같다.
# 
# - 입력 데이터셋: 2차원 좌표로 표현된 데이터셋
# - 타깃 데이터셋: 각 좌표의 점이 갖는 색깔
# - 모델 평가지표: 점의 색을 정확하게 예측한 비율
# 
# 점의 색깔을 예측하기 위한 한 가지 방식은 좌표의 축을 위 가운데 그림에서처럼 다르게 선택하는 것이다.
# 그러면 위 오른쪽 그림에서처럼 백점과 흑점을 단순히 $x$-좌표의 음수와 양수 여부에 따라 
# 간단하게 판단할 수 있다.
# 즉, 새로운 축을 사용하여 각 점의 좌표를 새롭게 표현하도록 변환하는 머신러닝 알고리즘을
# 구현하면 백점과 흑점을 매우 간단하게 선형적으로 분류하는 머신러닝 모델을 구현할 수 있다.
# :::

# **데이터 변환 자동화**
# 
# 위 예제의 경우 수동으로 데이터 변환 방식을 어렵지 않게 알아낼 수 있었다.
# 반면에 MNIST 손글씨의 경우처럼 각 숫자를 구분하는 쉽게 구분하는
# 데이터 변환을 수동으로 찾기는 거의 불가능하다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml2/master/slides/images/ch03/homl03-10.png" width="300"/></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.hanbit.co.kr/store/books/look.php?p_code=B9267655530">핸즈온 머신러닝(2판)</a>&gt;</div></p>

# 그런데 머신러닝 모델의 학습을 이용하면
# 데이터를 보다 유용한 방식으로 표현으로 변환을 자동으로 찾을 수 있다.
# 이때 사용되는 모델의 평가 기준은 **'새로운 데이터 표현법이
# 과제 해결에 보다 도움이 되는가'** 이며,
# 평가 기준에 따라 모델 학습에 사용되는 평가 지표가 지정된다.
# 
# 머신러닝 학습에 사용되는 대표적인 변환 방식은 다음과 같다.
# 
# - 픽셀 수 세기
# - 닫힌 원의 수 세기
# - 사영<font size='2'>projection</font>
# - 회전
# - 이동
# - 비선형 변환: $x > 0$ 의 데이터만 선택하기 등
# 
# 머신러닝 모델은 한 개 이상의 변환을 조합하며 과제에 따라 서로 다른 조합이 선택된다.

# **가설 공간**
# 
# 주어진 문제에 가장 적절한 변환을 머신러닝 알고리즘 스스로 알아내기는 기본적으로 불가능하다.
# 머신러닝 모델은 학습과정에서 사용할 수 있는 데이터 변환 방식을 지정하는 방식으로 구현되며,
# 이를 통해  머신러닝 알고리즘이 사용할 수 있는 변환 알고리즘의 공간을 제한한다.
# 
# 이와같이 머신러닝 모델이 주어졌을 때 모델 학습에 사용할 수 있는 알고리즘의 공간을
# **가설 공간**이라 부른다.
# 즉, 머신러닝 모델이 정해지면 모델은 
# **주어진 입력 데이터셋과 타깃 데이터셋을 이용하여 입력이 주어지면 타깃을 예측하고, 예측 성능을 평가지표을 이용하여 측정하는 과정을 반복 훈련**하면서
# 지정된 가설 공간 내에서 가장 적절한 데이터 변환법을 학습한다.

# ### 딥러닝

# **딥<font size='2'>deep</font>의 의미**

# 딥러닝의 딥<font size='2'>deep</font>은 
# 데이터 변환을 지원하는 **층**<font size='2'>layer</font>을 
# 세 개 이상 연속적으로 활용하는 머신러닝 모델 학습법을 가리킨다.
# 이런 의미에서 딥러닝을 **계층적 표현 학습법**이라 부르리도 한다.
# 반면에 **쉘로우 러닝**<font size='2'>shallow learning</font>는
# 한 두 개의 층만 사용하는 학습을 의미한다.

# **신경망 모델**
# 
# 딥러닝 모델은 **심층 신경망**<font size='2'>deep neural network</font>로 구성된다.
# 신경망은 여러 개의 층을 쌓아 올린 구조를 갖는다.
# 신경망의 깊이는 쌓아 올려진 층의 높이이며,
# 심층 신경망은 경우에 따라 수 십 또는 수 백 층으로 구성되기도 한다(아래 그림 참고).
# 모든 층에서 데이터 표현의 변환이 **자동**으로 이루어지는 것이 핵심이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml2/master/slides/images/ch14/homl14-15b.png" width="600"/></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html">ImageNet Classification with Deep Convolutional Neural Networks</a>&gt;</div></p>

# :::{prf:example} 손글씨 숫자 인식
# :label: exc-minist-transform
# 
# 아래 이미지는 네 개의 층을 사용하는 딥러닝 모델이
# MNIST 손글씨 이미지를 변환해서 최종적으로
# 이미지가 가리키는 숫자를 예측하는 과정을 보여준다.
# 
# 하나을 층을 지날 때마다 원본 이미지와 점차 많이 달라지는
# 방식으로 데이터를 표현하여 
# 최종적으로 (사람은 이해할 수 없지만)
# 모델은 쉽게 적절한 값을 예측할 수 있는 데이터로 변환한다.
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-mnist_representations.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>
# :::

# ### 딥러닝 작동 원리

# 딥러닝 모델이 사용하는 심층 신경망은
# 입력 데이터를 여러 층을 통과시키는 방식으로 변환시킨다.
# 딥러닝 모델의 심층 신경망의 핵심 요소는 다음 네 가지이다.
# 
# - 가중치
# - 손실 함수
# - 역전파
# - 훈련 루프

# **가중치**
# 
# 심층 신경망의 각 층은
# 데이터 변환에  **가중치**<font size='2'>weight</font>라 불리는
# 파라미터가 사용된다.
# 각 층은 입력되는 데이터를 층 고유의 가중치들과 조합하여
# 출력값을 생성한 후 다음 층으로 전달한다(아래 그림 참고).
# 
# 이와 같이 입력값을 가중치와 조합하는 과정을 여러 층을 통해 수행하여
# 최종 결과물인 예측값<font size='2'>prediction</font>를 
# 생성하는 과정을 **순전파**<font size='2'>feedforward</font>라 한다.
# 
# 머신러닝 모델의 학습은 모든 층에 대해 적절한 가중치를 찾는 과정을 의미한다.
# 경우에 따라 수 천만 개의 적절한 파라미터를 동시에 찾아야 하는
# 매우 어렵거나 불가능한 과제가 되기도 한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-deep-learning-in-3-figures-1.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **손실 함수**
# 
# 모델의 **손실 함수**<font size='2'>loss function</font>는
# 모델의 최종 출력결과인 예측값과 실제 타깃<font size='2'>target</font>이
# 얼마나 다른지를 측정한다.
# 손실 함수는 **목적 함수**<font size='2'>objective function</font> 또는 
# **비용 함수**<font size='2'>cost function</font>라고도 불린다.
# 
# 손실 함수가 반환값이 학습 과정에 있는 모델의 성능을 나타내면
# 기본적으로 0에 가까운 낮은 점수일 수록 잘 학습되었다고 말한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-deep-learning-in-3-figures-2.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **역전파**
# 
# **역전파**<font size='2'>backpropagation</font>는
# 경사하강법에 기초하여 학습 중인 모델의 손실 함수의 반환값을
# 최대한 낮추는 방향으로
# 각 층의 가중치를 조절하는 과정을 가리킨다.
# 역전파 알고리즘은 **옵티마이저**(optimizer)에 의해 실행되며
# 딥러닝 모델 학습의 핵심 역할을 수행한다.
# 
# 모델 훈련은 임의로 초기화된 가중치로 시작한다.
# 이후 옵티마이저가 각 층의 가중치를 손실값이 보다 작아지는 방향으로 조금씩 업데이트하는 
# **역전파 과정을 반복 실행**하면서 손실값이 점차 낮아져서 최종적으로 손실 최소값을 갖도록 하는
# 가중치를 찾아 간다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ch01-deep-learning-in-3-figures-3.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **훈련 루프**
# 
# 딥러닝 모델의 **훈련 루프**<font size='2'>training loop</font>는
# 순전파-손실값 계산-역전파의 구성된 하나의 순환과정을 가리킨다.
# 즉, 입력값을 받아 예측값을 생성한 후 손실값을 계산하여
# 그에 따라 가중치를 업데이트하는 일련의 과정이 훈련 루프이다.
# 
# 딥러닝 모델의 학습은 훈련 루프를 적게는 몇 십번, 많게는 몇 백, 몇 천 번 이상 
# 반복해야 최소 손실값을 갖는 가중치를 찾을 때까지 반복한다.

# <div align="center"><img src="https://github.com/codingalzi/dlp2/blob/master/jupyter-book/imgs/ch01-deep-learning-in-3-figures-3-a.png?raw=true" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 딥러닝의 성과와 전망

# 지난 10여년 동안 다양한 분야에서 다음과 같은 혁명적 성과가 딥러닝에 의해 이루어졌으며
# 거의 모든 산업 분야로 딥러닝의 활용이 확대되고 있다.
# 
# - 사람과 비슷한 수준의 이미지 분류, 음성 인식, 필기 인식, 자율 주행
# - 상당한 성능의 기계 번역, TTS(text-to-speech) 변환
# - 구글 어시스턴트, 아마존 알레사 등의 디지털 도우미
# - 향상된 광고 타게팅, 웹 검색
# - 자연어 질문 대처 능력
# - 초인류 바둑 실력(2013 알파고)

# 하지만 단기적으로 너무 높은 기대를 갖는 것은 위험하다.
# 그리고 실망이 너무 크면 AI에 대한 투자가 급속도로 줄어들어
# 1970년대의 1차, 1990년대의 2차 AI 겨울(AI winter)가 올 수도 있다.
# 2020년대 초반 현재 중요한 문제에 본격적으로 딥러닝이 적용되고 있지만 대중화는 아직이며
# 여전히 딥러닝의 능력을 평가하는 정도라고 할 수 있다.
# 그럼에도 불구하고 장기적으로 딥러닝이 가져올 가능성은 무궁무진하다고 말할 수 있으며
# 인간의 삶을 획기적으로 변화시킬 것으로 기대된다.

# ## 머신러닝 역사

# 산업계에서 사용되는 머신러닝 알고리즘의 대부분은 딥러닝 알고리즘이 아니다.
# 딥러닝 모델의 훈련에 필요한 데이터가 너무 적을 수도 있고,
# 아니면 딥러닝이 아닌 다른 알고리즘이 보다 좋은 성능을 발휘하기 때문이다.
# 따라서 딥러닝 모델이 아닌 다른 머신러닝 모델에 대해서도 잘 알고 있어야 한다.
# 
# 여기서는 딥러닝 발전 이전의 머신러닝의 역사를 간단하게 살펴본다.
# 아직도 많이 활용되는 머신러닝 모델에 대한 자세한 설명은
# [핸즈온 머신러닝(3판)](https://codingalzi.github.io/handson-ml3/intro.html)을
# 참고할 수 있다.

# **확률적 모델링**
# 
# 통계 법칙을 데이터분석에 적용하는 기법을 가리키며
# 지금도 많이 활용된다.
# 베이즈 정리<font size='2'>Bayes theorem</font>를 이용하는
# **나이브 베이즈 알고리즘**이 대표적이며
# 현재와 같은 컴퓨터가 없었던 1950년대부터 데이터분석에 활용되었다.
# 로지스틱 회귀 분류 모델이 나이브 베이즈 알고리즘과 유사하게 작동한다.

# **초창기 신경망**
# 
# 신경망의 기본 아이디어는 1950년대부터 연구되기 시작했다.
# 하지만 역전파를 어느 정도 제대로 실행할 수 있게 된 1990년 전후까지
# 제대로된 신경망 활용은 없었다.
# 최초의 성공적인 신경망 활용은 1989년 미국의 벨 연구소에서 이루어졌다.
# 얀 르쿤<font size='2'>Yann LeCun</font>이
# 손글씨 숫자 이미지를 자동으로 분류하는 시스템인 **LeNet 합성곱 신경망**을
# 소개했으며 1990년대에 미국 우체국에서 
# 우편번호를 자동 분류하는 데에 이용되었다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml2/master/slides/images/ch14/homl14-16.gif" width="400"/></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="http://yann.lecun.com/exdb/lenet/index.html">LeNet-T CNN</a>&gt;</div></p>

# **커널 기법**
# 
# LeNet 합성곱 신경망의 성공에도 불구하고 1990년대의 신경망 모델의 활용은 매우 제한적이었다.
# 그러다가 1995년에 소개된 **커널 기법**<font size='2'>kernel method</font>이
# 개발되면서 당시의 신경망 모델의 성능을 넘어서는 새로운 머신러닝 모델이 개발되었다.
# 
# 커널 기법을 사용하는 대표적인 머신러닝 모델이 바로 **서포트 벡터 머신**(SVM)이다.
# SVM은 비선형 방식으로 작동할 수 있는 분류 모델이며
# 딥러닝이 보다 발전하기 전까지 최고 성능의 모델로 활용되었다
# ([서포트 벡터 머신](https://codingalzi.github.io/handson-ml3/svm.html) 참고).
# SVM은 하지만 대용량 데이터셋 처리에 부적합하며, 무엇보다도
# 손글씨 인식, 이미지 분류 등 지각 문제 해결에 활용되기 어렵다.

# **결정트리, 랜덤 포레스트, 그레이디언트 부스팅**
# 
# 2000년대에 들어서면서 **결정트리**<font size='2'>decision tree</font>가
# 인기를 얻기 시작했다.
# 많은 수의 결정트리에 앙상블<font size='2'>ensemble</font> 기법을 적용한
# **랜덤 포레스트**<font size='2'>random forest</font>가 2010년 경에 소개되어
# 커널 기법보다 선호되기 시작했다.
# 
# 기존에 주어진 모델의 성능을 좀 더 향상시키는 
# **그레이디언트 부스팅**<font size='2'>gradient boosting</font> 기법이
# 2014년 소개되었으며
# 랜덤 포레스트의 성능을 뛰어 넘는 모델에 활용되었다.
# 현재까지도 딥러닝과 더불어 가장 많이 활용되는 기법중의 하나다.

# **딥러닝의 본격적 발전**
# 
# 2011년 GPU를 활용한 딥러닝 모델 훈련이 시작되었으며,
# 이미지 분류 경진대회인 [이미지넷의 ILSVRC](https://www.image-net.org/challenges/LSVRC/index.php)의
# 2012년 대회에서 이전 년도 우승 모델의 성능을 훨씬 뛰어 넘는 
# **합성곱 신경망**<font size='2'>convolutional neural network</font>(CNN) 모델이 소개되면서
# 딥러닝에 대한 관심이 폭발적으로 증가했다.
# 
# - 2011년 최고 모델의 성능: 74.3%의 top-5 정확도
# - 2012년 최고 모델의 성능: 83.6%의 top-5 정확도
# 
# 이미지넷 경진대회는 2015년 96.4%의 top-5 정확도 성능을 보인 우승 모델이
# 소개된 이후로 더 이상 진행되지 않는다.
# 이는 이미지 분류 과제가 완성되었음을 의미한다.
# 
# :::{admonition} ILSVRC와 top-5 정확도
# :class: info
# 
# 분류 모델의 성능을 평가할 때 top-5 정확도, top-1 정확도, top-5 오류율, top-1 오류율 등을 사용한다.
# ILSVRC 이미지 분류 경진대회는 1,400만 개 이상의 이미지를 1,000 개의 범주로 분류하는
# 모델을 평가한다. 
# 분류 모델은 각 이미지에 대해 이미지에 담긴 객체(사람, 사물, 품종 등)가 속하는 범주를
# 1,000개의 범주 전체에 대해 확률을 계산한다.
# 이때 가장 높은 확률을 가진 5개의 범주에 정답이 포함될 확률이 top-5 정확도이다.
# top-1 정확도는 가장 높은 확률을 갖는 범주가 정답일 확율을 의미한다. 
# 정답이 아닐 확률을 계산하면 top-5 오류율 또는 top-1 오류율이 계산된다.
# 
# top-1이 아닌 top-5를 모델 성능의 평가 기준으로 사용하는 이유는 경우에 따라
# 매우 유사한 범주가 많아 인간조차도 정확히 분류하기 어려울 수 있다는 점을 고려했기 때문이다. 
# 예를 들어, 고양이와 삵, 개와 늑대 등은 이미지로 쉽게 분류하기 어렵다.
# :::
# 
# 2015년 이후로 많은 영역에서 SVM, 결정트리 등 전통적인 머신러닝 모델이
# 딥러닝 모델로 대체되기 시작했다.
# 이유는 데이터에 내재되어 있지만 사람이 직접 알아내기는 매우 어렵거나 불가능한 
# 복잡한 구조를 딥러닝 모델은 데이터가 여러 층을 통과시키면서 스스로 찾아낼 수 있기 때문이다.
# 예를 들어 이미지와 영상을 분석하는 컴퓨터 비전 분야에서 CNN의 활용이 앞도적이다.
# 또한 번역, 통역, 문맥 이해 등 인식과 관련된 모든 분야에서
# 딥러닝 모델의 성능이 나날이 발전하고 있다.

# **최근 머신러닝 분야 동향**
# 
# 2019년 캐글(Kaggle) 경진대회에서 5등 이내 팀이 사용한 머신러닝 도구에 대한 설문조사 결과
# 케라스<font size='2'>Keras</font>와 텐서플로우<font size='2'>TensorFlow</font>의
# 조합이 가장 많이 사용되었다.
# 그 뒤를 이어 그레이디언트 부스팅 기법과 파이토치<font size='2'>PyTorch</font>의
# 인기도가 높다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/kaggle_top_teams_tools.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 지난 4년 동안의 머신러닝 도구의 인기도 변화도 비슷하다.

# <div align="center"><img src="https://github.com/codingalzi/dlp2/blob/master/jupyter-book/imgs/ch01-dlp2-kaggle-survey-ML-framework.png?raw=true" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://storage.googleapis.com/kaggle-media/surveys/Kaggle's%20State%20of%20Machine%20Learning%20and%20Data%20Science%202021.pdf">www.kaggle.com/kaggle-survey-2021(36쪽)</a>&gt;</div></p>

# 이미지 분류, 텍스트 분석, 음성 인식 등 지적 활동과 관련된 문제의 해결을 위해서는 케라스, 파이토치와 
# 같은 딥러닝 프레임워크가 일반적으로 사용되고,
# 그 이외의 경우엔 사이킷런과 XGBoost, LightGBM 등 그레이디언트 부스팅 기법의 조합이
# 많이 활용된다.

# ## 딥러닝 발전 동력

# 이미지 분석과 시계열 데이터 분석이 2012년 이후 획기적으로 발전하였지만
# 이런 획기적 발전에 기여한 아래 기법은 이미 1990년대에 제시되었다.
# 
# - 1990년: 합성곱 신경망과 역전파
# - 1997년: LSTM(Long Short-Term Memory)
# 
# 1990년대와 2010년대의 차이를 만든 요소는 다음 세 가지다.
# 
# - 하드웨어
# - 데이터
# - 알고리즘

# **하드웨어**
# 
# 다음 세 요소에서의 획기적 발전이 있었다.
# 
# - CPU: 1990년에서 2010년 사이에 5,000배 이상 빨라짐.
# - GPU(graphical processing unit): NVIDIA와 AMD가 2000년대부터 게임용 그래픽 카드 개발에 천문학적으로 투자 시작.
#     2007년 NVIDIA가 GPU를 프로그래밍에 활용할 수 있도록 해주는 CUDA 인터페이스 개발.
#     2011년부터 연구자들이 CUDA를 활용하는 신경망 개발 시작.
# - TPU(tensor processing unit): 2016년 구글이 소개한 딥러닝 전용 칩.
#     GPU보다 훨씬 빠르고 에너지 효율적임. 
#     2020년 최고의 슈퍼컴퓨터가 27,000 개의 NVIDIA GPU를 사용하는데 이는 대략 10개의 pod 성능에 맞먹음.
#     1 pod는 1024개의 TPU 카드를 의미함.
#     
# 이와 더불어 클라우드 컴퓨팅의 발전으로 하드웨어의 발전을 최대한 활용할 수 있게 되었다.

# **데이터**
# 
# 인터넷과 저장장치의 발전으로 인한 엄청난 양의 데이터를 축적하였다.
# YouTube(동영상), Flickr(이미지), Wikipedia(문서) 등이 컴퓨터 비전과 자연어 처리(NLP)의
# 혁신적 발전의 기본 전제조건이었다.

# **투자와 대중화**
# 
# 하드웨어, 데이터, 알고리즘 요소 이외에 투자와 대중화라는 두 요소 또한
# 딥러닝의 발전에 크게 기여하였다.
# 먼저, 투자 측면에서 보면 2013년 이후 인공지능 스타트업에 대한 투자가 획기적으로 증가하였다.
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/startup_investment_oecd.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.oecd-ilibrary.org/sites/3abc27f1-en/index.html?itemId=/content/component/3abc27f1-en&mimeType=text/html">OECD estimate of total investments in AI startups</a>&gt;</div></p>

# 그리고 대중화 측면에서는 지금은
# 파이썬 기초 프로그래밍 수준에서 딥러닝을 적용할 수 있을 정도로
# 많은 편리하며 뛰어난 성능의 도구와 프레임워크가 개발되었다.
# 
# - 사이킷런
# - 텐서플로우
# - 케라스
# - 파이토치 등등
