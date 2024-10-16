#!/usr/bin/env python
# coding: utf-8

# (ch:fundamentals_of_ml)=
# # 머신러닝 모델 훈련 기법

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 머신러닝 모델 훈련 기법](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-fundamentals_of_ml.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# 좋은 모델을 얻기 위해 알아야 할 기본 개념과 훈련법의 기초를 소개한다.
# 
# - 주요 개념: 일반화 대 최적화
# - 머신러닝 모델 평가 기법
# - 모델 훈련 최적화 기법
# - 모델 일반화 성능 향상 기법

# ## 머신러닝의 목표: 모델 일반화

# 훈련을 많이 할 수록 모델은 훈련 세트에 대해 보다 좋은 성능을 보이지만 새로운 데이터에 대한 
# 성능은 점점 떨어지는 과대적합 현상이 언제나 발생한다. 
# 머신러닝 모델 훈련의 주요 과제는 모델 훈련의 **최적화**(optimization)와 
# 모델 **일반화**(generalization) 사이의 관계를 적절히 조절하는 것이다.
# 
# - **최적화**: 훈련 세트에 대해 가장 좋은 성능을 이끌어 내는 과정
# - **일반화**: 처음 보는 데이터를 처리하는 모델의 능력
# 
# 문제는 일반화는 훈련을 통해 조절할 수 있는 대상이 아니라는 점이다.
# 하지만 모델 훈련 방식을 조정하면 일반화 성능을 끌어올릴 수 있다.

# **과소적합과 과대적합**

# - 과소적합
#     - 신경망이 훈련셋의 패턴을 아직 덜 파악한 상태
#     - 훈련셋과 검증셋 모두에 대해 성능이 향상되는 과정
#     - 보통 훈련 초반에 일어나는 현상.
#     - 모델 설정이 잘못된 경우에도 당연히 발생. 예를 들어, 비선형 분류 모델을 선형 모델로 훈련시키는 경우.
# 
# - 과대적합
#     - 잡음, 애매한 또는 매우 특별한 특성 등 훈련셋 고유의 패턴을 학습하기 시작하는 순간 발생
#     - 새로운 데이터와 무관하거나 혼동을 주는 패턴 학습

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/typical_overfitting.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **과대적합 발생 주요 요인**

# 과대적합을 발생시키는 요인은 크게 세 가지로 나뉜다.

# **_첫째, 훈련셋에 포함된 잡음_**

# 적절하지 않은 데이터 또는 잘못된 레이블을 갖는 데이터 등을 **잡음** 또는 **노이즈**<font size='2'>noise</font>라 부른다.
# 
# - 적절하지 않은 데이터: 다음 MNNIST 이미지들처럼 불분명하면 특성 파악이 어렵다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/fucked_up_mnist.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# - 잘못된 레이블: 예를 들어, 잘못 분류된 1처럼 생긴 이미지를 7로 잘못 분류할 가능성이 높아진다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/mislabeled_mnist.png" style="width:660px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 잡음 등의 이상치<font size='2'>outlier</font>를 학습하면
# 아래 오른편 그림의 경우처럼 모델이 이상치의 특별한 특성을 학습하게 되어
# 새루운 데이터에 대한 예측 성능이 불안정해지게 된다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/outliers_and_overfitting.png" style="width:660px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **_둘째, 애매한 특성_**

# 잡음 등의 이상치가 전혀 없다 하더라도 특정 특성 영역에 대한 예측값이 여러 개의 값을 가질 수 있다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/iris01.png" style="width:500px;"></div>

# 예를 들어, 붓꽃 데이터셋의 경우 꽃잎의 길이와 너비만을 활용해서는 
# 버시컬러<font size='2'>versicolor</font> 품종과 
# 버지니카<font size='2'>virginica</font> 품종의 완벽한 구분이
# 애초에 불가능하다.

# <div align="center"><img src="https://codingalzi.github.io/handson-ml2/slides/images/ch05/homl05-03b.png" style="width:500px;"></div>

# 하지만 훈련을 오래 시키면 각 샘플의 특성을 해당 레이블의 고유의 특성으로
# 간주하는 정도까지 모델이 훈련되어 아래 오른편 그림과 같이
# 샘플의 특성에 너무 민감하게 작동한다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/HighResolutionFigures/figure_5-5.png" style="width:660px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **_셋째: 특성과 타깃 사이의 거짓 상관관계_**

# 특성과 타깃 사이의 거짓 상관관계를 유발하는 여러 상황이 있다.

# - 매우 드문 특성을 사용하는 데이터셋으로 훈련하는 경우.
# 
# {numref}`%s절 <sec:imdb>`의 이진 분류 모델의 훈련에 사용된
# IMDB 데이터셋에서 매우 낮은 빈도로 사용되는 단어를 훈련셋에서 포함시키는 경우
# 어쩌다 한 번 사용되는 특성으로 인해 잘못된 판단이 유발될 수 있다.
# 
# 예를 들어, 에쿠아도르, 페루 등 안데스 산맥 지역에서 자라는 Cherimoya(체리모야) 라는
# 과일 이름이 들어간 영화 후기가 단 하나만 있으면서 마침 부정적이었다면,
# 분류 모델은 Cherimoya 단어가 들어간 영화 후기를 기본적으로 부정적으로 판단할 가능이 높아진다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/cherimoya.jpg" style="width:300px;"></div>
# 
# <p><div style="text-align: center">체리모야 열매</div></p>

# 이렇듯 매우 매우 드물게 사용되는 특성은 과대적합을 유발한다. 
# 앞서 사용 빈도에서 10,000등 안에 드는 단어만으로 작성된 영화 후기만을 대상으로 훈련시킨 이유가
# 이런 가능성을 제한하기 위해서였다.

# - 우연에 의한 경우.
# 
# 자주 발생하는 특성이라 하더라도 우연히 잘못된 편견을 훈련중인 모델에 심어줄 수 있다.
# 예를 들어, "너무" 라는 단어를 포함한 100개의 영화 후기 중에서 54%는 긍정,
# 나머지 46%는 부정이었다면 훈련중인 모델은 "너무"라는 단어를 긍정적으로 평가할 가능성을 높힌다.
# 하지만 "너무"라는 단어는 긍정적으로, 부정적으로 사용될 수 있기 때문에 이는 우연에 불과하다.

# - 의미 없는 특성에 의한 경우.
# 
# 아래 이미지는 MNIST 데이터셋에 **백색 잡음**<font size='2'>white noise</font>이 추가된 경우와
# 단순히 여백이 추가된 경우의 훈련 샘플을 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch05-mnist-noise.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 백색 잡음이 포함된 샘플들의 데이터셋으로 훈련시킨 모델과
# 단순한 여백이 추가된 샘플들의 데이터셋으로 훈련시킨 모델의 성능을 비교하면
# 백색 잡음이 포함된 훈련셋을 이용한 모델의 성능이 1% 정도 떨어진다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch05-mnist-noise03.png" style="width:400px;"></div>

# 이유는 모델이 백색 잡음에 특별한 의미를 부여하기 때문이며, 이로 인해 과대적합이
# 보다 쉽게 발생한다.
# 따라서 보다 효율적인 훈련을 위해 백색 잡음 등 과대적합을 유발하는
# 특성을 미리 제거하여 모델에 유용한 특성만을 훈련에 사용해야 한다.
# 하지만 유용한 특성을 선택하는 일이 기본적으로 불가능하거나 매우 어렵다.

# :::{admonition} 백색 잡음
# :class: info
# 
# 프로그래밍 분야에서 백색 잡음<font size='2'>white noise</font>은
# **무작위적으로 생성되었지만 전 영역에 걸쳐 고르게 퍼진 데이터셋**을 의미한다.
# 백색 잡음이 다른 데이터와 섞일 경우 백색 잡음을 분리하는 것은 원천적으로 불가능하다.
# 백색 잡음을 백색 소음 또는 화이트 노이즈라 부르기도 한다.
# :::

# ### 딥러닝 모델 일반화

# (딥러닝) 모델이 훈련 중에 보지 못한 완전히 새로운 데이터에 대해 예측하는 것을 
# **일반화**<font size='2'>generalization</font>라고 한다.
# 그런데 모델의 일반화 능력은 모델의 훈련 과정에 별 상관이 없다.
# 이유는 훈련을 통해 모델의 일반화 능력을 조절할 수 있는 방법이 없기 때문이다.

# 실제로 딥러닝 모델은 어떤 무엇도 학습할 수 있다.
# 예를 들어 MNIST 모델을 임의로 섞은 레이블과 함께 훈련시키면
# 훈련셋에 대한 성능은 훈련하면서 계속 향상되어 결국
# 모델은 모든 답을 외워버리는 정도에 다달한다.
# 물론 검증셋에 성능은 전혀 향상되지 않는다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch05-mnist-shuffled.png" style="width:400px;"></div>

# 위 결과는 다음을 의미한다.
# 
# - 일반화는 모델 훈련 과정 중에 제어할 수 있는 대상이 아니다. 
# - 모델 훈련을 통해 할 수 있는 것은 주어진 훈련 데이터셋에 모델이 적응하도록 하는 것 뿐이다.
# - 딥러닝 모델은 어떤 데이터셋에도 적응할 수 있기에 
#     너무 오래 훈련시키면 과대적합은 반드시 발생하고 일반화는 어려워진다. 
# - 모델의 **일반화** 능력은 모델 자체보다는 **훈련셋에 내재하는 정보의 구조**와 
#     보다 밀접히 관련된다.

# **다양체 가설과 모델 훈련**

# **다양체 가설**<font size='2'>manifold hypothesis</font>은 
# 데이터셋에 내재하는 정보의 구조에 대한 다음과 같은 가설이다.
# 
# > 일반적인 데이터셋은 고차원상에 존재하는 (저차원의) 연속이며 미분가능한 다양체를 구성한다.

# 예를 들어, 아래 이미지는 3차원 공간에 존재하는 2차원 다양체를 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch05-homl3-manifold.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://github.com/ageron/handson-ml2">핸즈온 머신러닝(2판), 8장</a>&gt;</div></p>

# 다양체 가설을 인정하면 모델 훈련은 다양체 형식으로 훈련셋에 내재된 정보의 구조를 
# 찾아가는 과정이 된다.
# 그리고 적절히 훈련된 모델이 완전히 새로운 데이터에 대해 적절한 예측을 할 수 있는 이유를 
# 다음과 같이 설명할 수 있다.
# 
# > 모델이 찾아낸 훈련셋에 내제된 정보의 구조가 부드러운 곡선을 갖는 다양체이기에 보간법을 적절히 적용할 수 있기 때문이다.

# 모델의 훈련 과정에서 훈련셋에 내재된 다양체를 찾는 과정은 
# 경사하강법을 이용하여 단계적으로 아주 천천히 진행된다.
# 따라서 훈련 진행 과정 중에 특정 단계에서 데이터셋에 내재된 다양체에
# 근접하고, 훈련이 더 진행될 수록 아래 그림에서처럼 
# 찾아야 하는 다양체에서 점차 멀어지는 과대적합 단계로 접어든다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/05-10.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 따라서 모델의 훈련을 적절한 단계에서 멈추도록 하면 모델의 일반화 성능을 최대한
# 끌어 올릴 수 있다. 아래에서 모델의 훈련을 적절할 때에 조기종료하는 방법을 살펴볼 것이다.

# **보간법과 일반화**

# **보간법**<font size='2'>interpolation</font>은 
# 모델 훈련에 사용된 훈련셋의 데이터와 새로운 데이터를 연결하는 
# 다양체 상의 경로를 이용하여 예측값을 결정하는 방법이다.
# 새로운 데이터에 대한 예측, 즉 모델의 일반화는
# 바로 이 보간법을 이용한다.
# 
# 아래 그림은 훈련셋이 작은 경우와 큰 경우의 차이를 잘 보여준다. 
# 훈련셋이 충분히 크지 않으면 모델이 찾아낸 다양체와 훈련셋에 내재된 실제 다양체 사이의
# 편차가 크기에 새로운 데이터에 대한 예측이 보다 부정확할 수밖에 없다.
# 반면에 양질의 훈련 데이터 샘플이 충분히 많다면 훈련을 통해 실제 다양체에 매우 근접하는 모델을
# 얻을 수 있게 된다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/dense_sampling.png" style="width:660px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **충분치 않은 훈련셋과 정규화**
# 
# 일반적으로 양질의 많은 데이터를 이용한 모델의 일반화 성능이 좋다.
# 하지만 [**차원의 저주**](https://codingalzi.github.io/handson-ml3/dimensionality_reduction.html)로 인해 충분한 크기의 훈련셋을 구하기가 일반적으로 불가능하거나 매우 어렵다. 
# 
# 충분히 큰 훈련셋을 준비하지 못하는 경우 **규제**<font size='2'>regularization</font>를 
# 이용해서 모델이 과대적합되는 것을 예방할 수 있다.
# 규제에 대해서 잠시 뒤에 자세히 다룬다.

# ## 모델 평가

# 모델의 일반화 능력을 향상시키려면 주어진 모델의 일반화 능력을 평가할 수 있어야 한다.

# ### 훈련셋, 검증셋, 테스트셋

# 훈련셋, 테스트셋 이외에 검증셋을 사용해야 하는 이유는 
# 무엇보다도 최적의 모델을 구성할 때 검증셋에 대한 결과가 반영되기 때문이다. 
# 
# 테스트셋은 모델 구성과 훈련에 전혀 관여하지 않아야 한다.
# 따라서 구성된 모델의 성능을 평가하려면 테스트셋을 제외한 다른 데이터셋이 필요하고
# 이를 위해 훈련셋의 일부를 검증셋으로 활용한다.
# 
# 검증셋은 **훈련 과정 중에 일반화 성능을 테스트**하는 용도로 사용되며
# 이를 통해 레이어 종류 및 개수, 레이어 별 유닛 개수 등 모델 구성에 필요한
# **하이퍼파라미터**<font size='2'>hyperparameter</font>를 조정한다. 
# 이것을 **모델 튜닝**<font size='2'>model tuning</font>이라 하며,
# 바로 이 모델 튜닝을 위해 검증셋이 사용되는 것이다. 

# **모델 튜닝과 정보 유출**

# 모델 튜닝도 모델의 좋은 하이퍼파라미터를 찾아가는 일종의 **학습**이다. 
# 따라서 튜닝을 많이 하게되면 검증셋에 대한 과대적합이 발생한다.
# 다시 말해, 검증셋에 특화된 튜닝을 하게 되어 모델의 일반화 성능이 떨어지게 된다. 
# 이런 현상을 **정보 유출**이라 하는데,
# 이유는 튜닝을 하면 할 수록 검증셋에 대한 보다 많은 정보가 모델로 흘러 들어가기 때문이다.

# :::{admonition} 하이퍼파라미터 대 파라미터
# :class: info
# 
# 하이퍼파라미터와 파라미터는 다르다. 
# 파라미터는 모델 훈련 중에 학습되는 가중치, 편향 등을 가리킨다.
# :::

# **검증셋 활용법**

# 데이터셋을 훈련셋, 검증셋, 테스트셋으로 분류하여 모델 훈련을 진행하는
# 전형적인 방식 세 가지를 소개한다. 

# *홀드아웃<font size='2'>hold-out</font> 검증*

# 훈련셋의 일부를 검증셋으로 사용하는 가장 일반적인 방법이며,
# 모델 훈련 후에 테스트셋을 이용하여 모델의 일반화 성능을 확인한다.
# 하지만 그 이후엔 더 이상 모델 튜닝을 진행하지 않아야 한다. 

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/holdout_validation.png" style="width:350px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 홀드아웃 검증은 하지만 데이터셋이 크지 않은 경우 사용하기 어렵다.
# 이유는 데이터셋 자체가 작기 때문에 훈련셋과 검증셋은 더욱 작아져서
# 제대로된 훈련이 이뤄지기 어려울 수 있기 때문이다.
# 이런 경우 K-겹 교차검증이 권장된다.

# *K-겹 교차검증*

# 훈련셋을 K개의 부분집합으로 분류한 다음에 한 개의 부분집합을 검증셋으로,
# 나머지늘 훈련셋으로 사용하는 방식을 K 번 반복하는 훈련법이다.
# 이때 검증셋으로 사용되는 부분집합은 매번 다르게 선택된다.
# 모델의 성능은 K개의 검증성능의 평균값으로 평가된다.
# 
# 아래 그림은 3-겹 교차검증의 작동 과정을 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/k_fold_validation.png" style="width:650px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# *반복 K-겹 교차검증*

# 훈련셋의 크기가 너무 작거나 모델의 성능을 최대한 정확하게 평가하기 위해 사용된다.
# K-겹 교차검증을 여러 번 실행한다. 대신 매번 훈련셋을 무작위로 섞은 뒤에
# 교차검증을 실행한다. 
# 최종 결과는 각 교차검증의 평균값을 사용한다. 
# 훈련 시간이 매우 오래 걸린다는 게 이 방법의 단점이다. 
# K-겹 교차 검증을 P번 반복하면 총 `P * K` 개의 모델을 훈련시키게 된다.

# ### 모델 성능 평가의 기준선

# 모델 훈련이 시작되면 평가지표<font size='2'>metrics</font>를 지켜보는 일 이외에 할 수 있는 게 없다.
# 다만 훈련중에 검증셋에 대한 평가지표가 특정 기준선 이상인지를 아는 것은 매우 중요하다.
# 
# 앞서 살펴봤던 모델들의 기준선은 다음과 같다.
# 
# - MNIST 데이터셋: 10%의 정확도
# - IMDB 데이터셋: 50%의 정확도
# - 로이터 통신 기사: 18-19%의 정확도. 기사들이 균등하게 분포되어 있지 않음.
# 
# 기준선을 넘는 모델을 학습을 통해 얻는 것이 기본 목표이어야 한다.
# 그렇지 않다면 무언가 잘못된 모델을 또는 잘못된 접근법을 사용하고 있을 가능성이 크다.

# ### 모델 평가용 데이터셋 준비 관련 주의사항

# 최적화된 모델 훈련을 위해 아래 세 가지 사항을 준수하며 훈련셋, 검증셋, 테스트셋을 
# 준비해야 한다. 
# 
# - **대표성**: 일반적으로 데이터셋을 무작위로 섞어 레이블이 적절한 비율로 섞인 
#     훈련셋, 검증셋, 테스트셋을 구성해야 한다.
# - **순서 준수**: 미래를 예측하는 모델을 훈련시킬 때, 테스트셋의 데이터는 훈련셋의 데이터보다
#     시간상 뒤쪽에 위치하도록 해야 한다. 그렇지 않으면 미래 정보가 모델에 유출된다.
#     즉, 데이터를 무작위로 섞어 훈련셋과 테스트셋으로 구분하는 일은 하지 않아야 한다. 
# - **중복 데이터 제거**: 훈련셋과 테스트셋에 동일한 데이터가 들어가지 않도록 중복 데이터를 제거해야 한다.
#     그렇지 않으면 공정한 검증이 이뤄질 수 없다.

# ## 모델 훈련 개선법

# 모델 훈련을 최적화하려면 먼저 과대적합을 달성해야 한다.
# 이유는 과대적합이 나타나야만 과대적합의 경계를 알아내서 모델 훈련을
# 언제 멈추어야 하는지 알 수 있기 때문이다. 
# 일단 과대적합의 경계를 찾은 다음에 일반화 성능을 목표로 삼아야 한다.
# 
# 모델 훈련 중에 발생하는 문제는 크게 세 종류이다. 
# 
# - 첫째, 훈련셋의 손실값이 줄어들지 않아 훈련이 제대로 진행되지 않는 경우
# - 둘째, 훈련셋의 손실값은 줄어들지만 검증셋의 성능은 평가 기준선을 넘지 못하는 경우
# - 셋째, 훈련셋과 검증셋의 평가가 기준선을 넘어 계속 좋아지지만 과대적합이 발생하지 않아
#     계속해서 과소적합 상태로 머무는 경우

# **첫째 경우: 경사하강법 관련 파라미터 조정**

# 훈련셋의 손실값이 줄어들지 않거나 진동하는 등 훈련이 제대로 이루어지지 않는 경우는
# 기본적으로 경사하강법이 제대로 작동하지 않아서이다.
# 이럴 때는 보통 학습률과 배치 크기를 조절하면 해결된다.

# *학습률 조정*

# - 매우 큰 학습률을 사용하는 경우: 모델이 제대로 학습되지 않는다.

# <div align="center"><img src="https://codingalzi.github.io/handson-ml2/slides/images/ch04/homl04-03.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://github.com/ageron/handson-ml2">핸즈온 머신러닝(2판), 4장</a>&gt;</div></p>

# - 매우 작은 학습률을 사용하는 경우: 모델이 너무 느리게 학습된다.

# <div align="center"><img src="https://codingalzi.github.io/handson-ml2/slides/images/ch04/homl04-02.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://github.com/ageron/handson-ml2">핸즈온 머신러닝(2판), 4장</a>&gt;</div></p>

# - 적절한 학습률을 사용하는 경우: 모델이 적절한 속도로 학습된다.

# <div align="center"><img src="https://codingalzi.github.io/handson-ml2/slides/images/ch04/homl04-01.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://github.com/ageron/handson-ml2">핸즈온 머신러닝(2판), 4장</a>&gt;</div></p>

# *배치 크기 조정*

# 학습률 이외에 배치크기를 키우면 보다 안정적으로 훈련이 된다.
# 단, 계산량이 많아져서 훈련 시간이 훨씬 오래걸리게 된다.
# 아래 그림은 선형회귀 모델 훈련에 사용된 파라미터가 배치 크기가 클 수록 보다 
# 부드럽게 특정 값에 수렴하는 것, 즉 모델이 보다 일관되게 학습되는 것을 보여준다.

# <div align="center"><img src="https://codingalzi.github.io/handson-ml2/slides/images/ch04/homl04-05.png" style="width:550px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://github.com/ageron/handson-ml2">핸즈온 머신러닝(2판), 4장</a>&gt;</div></p>

# **둘째 경우: 보다 적절한 모델 사용**

# 훈련은 잘 진행되는데 검증셋에 대한 성능이 좋아지지 않는다면 다음 두 가지 경우를 의심해 보아야 한다. 
# 
# - 훈련셋이 적절한지 않은 경우
#     - 예제: 레이블이 무작위로 섞인 MNIST 데이터셋
# - 사용하는 모델이 적절하지 않은 경우
#     - 예제: 선형 분류가 불가능한 데이터셋에 선형분류 모델을 적용하는 경우
#     - 예제: 시계열 데이터 분석에 앞서 살펴본 `Sequential` 모델을 사용하는 경우
#     
# 문제 해결에 적절한 모델을 훈련시켜야 한다.
# 앞으로 다양한 문제에 적용되는 다양한 모델 구성과 훈련을 살펴볼 것이다.

# **셋째 경우: 모델의 정보 저장 능력 조정**

# 모델의 훈련셋/검증셋의 평가지표가 계속 향상되지만 과대적합이 발생하지 않는 경우 
# 기본적으로 모델의 정보 저장 능력을 키워야 한다. 
# 즉, 신경망 모델의 은닉층 또는 층에 사용되는 유닛의 수를 증가시켜서
# 모델이 보다 많은 정보를 처리하며 보다 적절한 데이터 표현의 변환이 가능하도록 해야 한다.

# ## 일반화 향상법

# 모델 훈련이 어느 정도 잘 진행되어 일반화 성능이 향상되고 과대적합이 발생하기 시작하면
# 모델의 일반화 성능을 극대화하는 방법에 집중해야 한다.
# 
# 일반적으로 다음 네 가지 사항이 모델의 일반화 성능에 도움을 준다.
# 
# - 양질의 데이터셋
# - 특성 조종
# - 조기 종료
# - 규제

# ### 양질의 데이터셋

# 양질의 데이터셋을 모델 훈련에 사용해야 모델이 
# 데이터셋에 잠재되어 있는 정보의 특성을 보다 잘 찾아낼 수 있다. 
# 양질의 데이터를 보다 많이 수집하는 일이 보다 적절한 모델을 찾으려는 노력보다
# 값어치가 높다. 
# 
# 양질의 데이터는 다음 기준을 만족시켜야 한다.
# 
# - 충분한 양의 샘플. 훈련셋이 크면 클 수록 일반화 성능이 좋아짐.
# - 타깃(레이블) 지정 오류 최소화.
# - 적절한 데이터 전처리. 데이터 클리닝과 결측치 처리.
# - 유용한 특성 선택. 주요 특성에 집중하면 훈련 시간도 줄이면서 동시에 
#     보다 높은 일반화 성능의 모델을 훈련시킬 수 있음.

# ### 특성 조종

# 신경망 모델은 데이터셋에 잠재되어 있는 정보의 특성을 보다 쉽게 알아 내기 위해 
# 여러 층을 통해 데이터셋의 표현을 다양한 방식으로 변환시킨다.
# 
# 이때 유용한 특성으로 구성된 훈련셋으로 모델을 훈련하면 보다 적은 양의 데이터로
# 보다 효율적인 훈련이 가능하다.
# 이처럼 모델 훈련에 가장 유용한 특성으로 구성된
# 데이터셋을 준비하는 과정이 **특성 조종**(feature engineering)이며,
# **특성 공학**이라고 많이 불린다.

# 예를 들어 아래 그림은 현재 시간을 알아내는 모델을 학습시키려 할 때
# 입력 데이터로 사용될 수 있는 세 가지 형식을 보여준다.
# 
# - 첫째 형식: 시계 사진
# - 둘째 형식: 큰/작은 시계 바늘의 직교좌표
# - 셋째 형식: 큰/작은 시계 바늘의 극좌표

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/05-16.png" style="width:400px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 입력 데이터셋의 형식에 따라 현재 시간을 알아내는 프로그램을 구현하는 방식이 많이 달라질 수 있다.
# 
# - 첫째 형식: 합성곱 신경망<font size='2'>convolutional neural network</font>(CNN) 모델 활용. 
#     많은 양의 훈련 데이터 필요.
# - 둘째 형식: 시계 사진에서 각 바늘의 좌표를 알아낸 후 간단한 머신러닝 모델 활용 가능.
# - 셋째 형식: 각 바늘의 극좌표를 구한 후에 간단한 수식으로 시간 확인 가능. 머신러닝 모델 불필요.

# ### 조기 종료

# 모델 훈련 중에 훈련셋에 대한 성능은 계속 좋아지지만 
# 검증셋에 대한 성능이 더 이상 좋아지지 않는 순간에 모델 훈련을 멈추는 것이 
# **조기 종료**(early stopping)이다.
# 이를 위해 에포크마다 모델의 성능을 측정하여 가장 좋은 성능의 모델을 기억해 두고,
# 더 이상 좋아지지 않으면 그때까지의 최적 모델을 사용하도록 한다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch05-early-stopping.png" style="width:600px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 케라스의 경우 `EarlyStopping` 이라는 
# **콜백**<font size='2'>callback</font> 기능을 사용하면 조기 종료 기능을
# 자동으로 수행한다. 
# 다양한 콜백 기능에 대해서는 {numref}`%s장 <ch:working_with_keras>` 자세히 다룬다.

# ### 규제

# 모델이 훈련셋에 너무 익숙해지지 않도록 방해하는 것을
# **규제**<font size='2'>regularization</font>라 부른다.
# 규제를 통해 모델이 훈련셋에 포함되지 않은 데이터에 대해 보다 잘 예측하도록,
# 즉, 모델의 일반화 성능을 높힌다.
# 
# 모델 규제에 가장 많이 사용되는 기법 세 가지를 소개한다.

# **규제 기법 1: 신경망 크기 축소**

# 신경망 모델에 사용되는 층과 각 층에 포함된 유닛의 수를 줄여 모델을 단순화하면 
# 모델이 처리할 수 있는 정보량이 줄어들다.
# 결국 데이터 특성의 세세한 정보 보다는 가장 핵심적인 정보를 활용하도록 
# 훈련되어 훈련셋에 덜 특화된, 반면에 일반화 성능은 향상된 모델이
# 훈련될 수 있다.

# 아래 그래프는 {numref}`%s장 <ch:getting_started_with_neural_networks>`에 다룬
# IMDB 영화 후기 분류 모델 대상으로
# 신경망의 크기를 줄이면
# 과대적합이 보다 늦게 발생하고, 검증셋에 대한 손실값이 줄어드는 것을 보여준다.
# 즉, 모델의 일반화 성능이 좋아진다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/original_model_vs_smaller_model_imdb.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 아래 그래프는 IMDB 영화 후기 분류 모델 대상으로
# 신경망의 크기를 아주 크게 하면
# 과대적합이 매우 빠르게 발생하며, 검증셋에 대한 성능이 매우 불안정해질 수도
# 있음을 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/original_model_vs_larger_model_imdb.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 물론 모델이 너무 단순하면 아예 훈련이 제대로 되지 않을 수 있어서
# 적절한 수의 층과 유닛의 수를 찾아야 한다.
# 하지만 이에 대한 이론적 기준은 없으며 실험을 통해 적절한 수준을 정할 수 밖에 없다.
# 보통 적은 수의 층과 유닛을 사용하다가 점차 수를 늘려 나가는 방식을 택하는 것이 좋다.

# **규제 기법 2: 가중치 규제**

# **가중치 규제**<font size='2'>weight regularization</font>는
# 모델이 학습하는 파라미터(가중치와 편향)가 너무 큰 값을 갖지 않도록 유도한다.
# 두 가지 방식이 있다.
# 
# - L1 규제: 가중치들의 절댓값이 작아지도록 유도하며,
#     라쏘<font size='2'>Lasso</font> 규제라고도 불린다.    
# - L2 규제: 가중치들의 제곱이 작은 값을 갖도록 유도하며,
#     릿지<font size='2'>Ridge</font> 규제라고도 불린다.

# :::{admonition} 규제 적용
# :class: hint
# 
# 규제는 훈련 중에만 적용되며 실전에는 사용되지 않는다.
# :::

# 규제는 층 단위로 지정되며 아래 형식을 따른다.
# 
# ```python
# layers.Dense(16, 
#              kernel_regularizer=regularizers.l2(0.002), 
#              activation="relu")
# ```
# 
# 위 코드는 L2 규제를 사용하는 층을 지정한다.
# 0.002는 규제 강도를 나타내며, 클 수록 강한 강도를 의미한다.
# 
# L1 규제 또는 L1 규제와 L2 규제를 함께 사용하려면 다음과 같이 규제를 지정한다.
# 
# ```python
# layers.Dense(16, 
#              kernel_regularizer=regularizers.l1(0.001), 
#              activation="relu")
# ```
# 
# 또는
# 
# ```python
# layers.Dense(16, 
#              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.002), 
#              activation="relu")
# ```

# :::{admonition} 릿지 회귀, 라쏘 회귀, 엘라스틱 넷
# :class: info
# 
# L1 규제, L2 규제, L1-L2 규제의 정확한 작동방식은 
# 각 규제가 적용된 선형회귀 모델인 
# [릿지 회귀, 라쏘 회귀, 엘라스틱 넷](https://codingalzi.github.io/handson-ml3/training_models.html)에 
# 대한 설명을 참고하면 된다.
# :::

# 아래 그래프는 IMDB 영화 후기 분류 모델 대상으로
# L2 규제를 가했을 때 검증셋에 대한 손실값이 적절하게 유지됨을 보여준다.
# 즉, 일반화 성능이 좋아진다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/original_model_vs_l2_regularized_model_imdb.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **규제 기법 3: 드롭아웃 적용**

# 가중치 규제 기법은 복잡한 딥러닝 모델에 대해서는 잘 작동하지 않는다.
# 심층 신경망 모델의 규제는 보통 드롭아웃 기법을 적용한다.
# **드롭아웃**은 무작위로 선택된 일정한 비율의 유닛을 끄는 것을 의미하며,
# 해당 유닛에 저장된 값을 0으로 처리한다. 
# 아래 그림은 50%의 드롭아웃을 적용했을 때의 가중치 행렬의 변화를 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/05-20.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 케라스에서는 적절한 드롭아웃 비율을 답은 드롭아웃 층 `Dropout`을 활용한다.
# 아래 그래프는 IMDB 영화 후기 분류 모델 대상으로 50%의 드롭아웃을 적용하여 훈련하면
# 검증셋에 대한 손실값이 보다 늦게, 보다 약하게 증가함을 보여준다. 
# 즉, 모델의 과대적합이 보다 늦게, 보다 약하게 발생한다.
# 

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/original_model_vs_dropout_regularized_model_imdb.png" style="width:500px;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ## 연습문제

# 1. [(실습) 머신러닝 모델 훈련 기법](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/excs/exc-fundamentals_of_ml.ipynb)
