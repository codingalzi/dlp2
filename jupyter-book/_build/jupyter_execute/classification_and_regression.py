#!/usr/bin/env python
# coding: utf-8

# (ch:classification_and_regression)=
# # 분류와 회귀

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 분류와 회귀](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-classification_and_regression.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# 간단한 실전 예제을 이용하여 신경망 모델을 
# 이진 분류, 다중 클래스 분류, 회귀 문제에 적용하는 방법을 소개한다.
# 
# - 이진 분류: 영화 리뷰 분류
# - 다중 클래스 분류: 뉴스 기사 분류
# - 회귀: 주택 가격 예측
# 
# 이를 통해 데이터 전처리, 모델 구조 설계, 모델 평가의 딥러닝 모델 훈련의
# 전체 과정을 자세히 살펴본다.

# **머신러닝 주요 용어**
# 
# 아래 용어의 정의를 명확히 알아야 한다.
# 
# | 한글 | 영어 | 뜻 |
# | :--- | :--- | :--- |
# | 샘플, 입력값 | sample, input | 모델 훈련에 사용되는 데이터 |
# | 예측값, 출력값 | prediction, output | 모델이 계산한 결과 |
# | 타깃 | target | 예측해야 하는 값 |
# | 예측 오류, 손실값 | prediction error, loss value | 타깃과 예측값 사이의 거리 측정값. 측정 방식에 의존함.|
# | 클래스 | class | 분류 문제에서 샘플이 속하는 범주 |
# | 레이블 | label | 분류 문제에서 타깃 대신 사용 |
# | 실제의/정답의 | ground-truth | 실제 조사 결과와 관련된 |
# | 이진 분류 | binary classification | 샘플을 두 개의 클래스로 분류. 양성/음성, 긍정/부정 등. |
# | 다중 클래스 분류 | multiclass classification | 샘플을 세 개 이상의 클래스로 분류. 손글씨 숫자 등. |
# | 다중 레이블 분류 | multilabel classification | 샘플에 두 종류 이상의 레이블을 지정하는 분류.사진 속 여러 마리 동물 등. |
# | 스칼라 회귀 | scalar regression | 샘플 별로 하나의 실숫값 예측하기. 주택 가격 예측 등. |
# | 벡터 회귀 | vector regression | 샘플 별로 두 개 이상의 실숫값 예측하기. 네모 상자의 좌표 등. |
# | 미니배치 | mini-batch | 보통 8개에서 128개의 샘플로 구성된 묶음(배치). 훈련 루프의 스텝 지정에 활용됨. |

# ## 영화 리뷰 분류: 이진 분류

# 영화 리뷰가 긍정적인지 부정적인지를 판단하는 이진 분류 모델을 구성한다.

# **IMDB 데이터셋**

# 긍정 리뷰와 부정 리뷰 각각 25,000개씩 총 50,000개의 영화 리뷰 샘플로 구성된 데이터셋이며,
# [IMDB(Internet Moview Database)](https://www.imdb.com/) 영화 리뷰 사이트에서
# 추출된 데이터로 케라스가 자체적으로 제공한다. 

# **IMDB 데이터셋 불러오기**

# 케라스의 `imdb` 모듈의 `load_data()` 함수로 데이터를 불러온다.
# 데이터셋이 훈련셋과 테스트셋으로 이미 구분되어 있다.
# 
# ```python
# from tensorflow.keras.datasets import imdb
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# ```
# 
# `num_words=10000` 키워드 인자는
# 가장 많이 사용되는 10,000개의 단어로만 구성된 리뷰를 불러오도록 지정한다.
# 10,000개의 단어에 포함되지 않는 단어는 무시한다.
# 
# 리뷰 전체에서 원래 총 88,585개의 단어가 최소 한 번 이상 사용되지만 가장 많이 사용되는
# 10,000개 단어 이외는 사용 빈도가 너무 낮아서 클래스 분류에 거의 도움되지 않는다.
# 따라서 그런 단어들은 무시하는 것이 좋다.

# 샘플들의 크기는 서로 다르다.

# In[1]:


len(train_data[0])


# In[7]:


len(train_data[1])


# 0번 샘플의 처음 10개 값은 다음과 같다.

# In[8]:


train_data[0][:10]


# 각 샘플의 레이블은 0(부정) 또는 1(긍정)이다.

# In[9]:


train_labels[0]


# In[10]:


test_labels[0]


# **리뷰 내용 확인하기**
# 
# *주의사항: 모델 훈련을 위해 반드시 필요한 사항은 아님!*

# - 정수와 단어 사이의 관계를 담은 사전 객체 가져오기

# In[11]:


word_index = imdb.get_word_index()


# `word_index`에 포함된 10개 항목을 확인하면 다음과 같다.

# In[12]:


for item in list(word_index.items())[:10]:
    print(item)


# In[13]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# `reverse_word_index`에 포함된 10개 항목을 확인하면 다음과 같다.

# In[14]:


for item in list(reverse_word_index.items())[:10]:
    print(item)


# 첫째 리뷰 내용을 아래와 같이 확인할 수 있다.
# 
# - 단어 인덱스에서 3을 빼야 함. 
# - 인덱스 0, 1, 2는 각각 여백, 문장 시작, 불분명을 의미함.

# In[15]:


first_review = train_data[0]

decoded_review = " ".join(
    [reverse_word_index.get(i-3, "?") for i in first_review])

decoded_review


# ### 데이터 전처리: 벡터화

# 정수들의 리스트, 그것도 길이가 다른 여러 개의 리스트를 신경망의 입력값으로 사용할 수 없다. 
# 또한 모든 샘플의 길이를 통일시킨다 하더라도 정수들의 리스트를 직접 신경망 모델의 입력값으로 
# 사용하려면 나중에 다룰 `Embedding` 층(layer)과 같은 전처리 층을 사용해야 한다. 
# 여기서는 대신에 **멀티-핫-인코딩**을 이용하여 정수들의 리스트를
# 0과 1로만 이루어진 일정한 길이의 벡터(1차원 어레이)로 변환한다. 

# #### 멀티-핫-인코딩

# 앞서 보았듯이 리뷰 리스트에 사용된 숫자들은 1부터 9999 사이의 값이다.
# 이 정보를 이용하여 리뷰 샘플을 길이가 10,000인 벡터(1차원 어레이)로 변환할 수 있다.
# 
# - 어레이 길이: 10,000
# - 항목: 0 또는 1
# - 리뷰 샘플에 포함된 정수에 해당하는 인덱스의 항목만 1로 지정
# 
# 예를 들어, `[1, 18, 13]`은 길이가 10,000인 1차원 어레이(벡터)로 변환되는데
# 1번, 18번, 13번 인덱스의 항목만 1이고 나머지는 0으로 채워진다.
# 이러한 변환을 **멀티-핫-인코딩**(multi-hot-encoding)이라 부른다.
# 다음 `vectorize_sequences()` 함수는 앞서 설명한 멀티-핫-인코딩을 
# 모든 주어진 샘플에 대해 실행한다.
# 
# 이처럼 각 샘플을 지정된 크기의 1차원 어레이로 변환하는 과정을 **벡터화**(vectorization)이라 한다.

# In[16]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):    # 모든 샘플에 대한 멀티-핫-인코딩
        for j in sequence:
            results[i, j] = 1.
    return results


# 이제 훈련셋와 테스트셋를 벡터화한다.

# In[17]:


x_train = vectorize_sequences(train_data).astype("float32")
x_test = vectorize_sequences(test_data).astype("float32")


# 첫째 훈련 샘플의 변환 결과는 다음과 같다.

# In[18]:


x_train[0]


# 레이블 또한 정수 자료형에서 `float32` 자료형으로 변환해서 자료형을 일치시킨다.

# In[19]:


y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


# ### 모델 구성

# 입력 샘플의 특성이 벡터(1차원 어레이)로 주어지고 
# 레이블이 스칼라(하나의 숫자)로 주어졌을 때 
# 밀집층(densely-connected layer)과
# `Sequential` 모델을 이용하면 성능 좋은 모델을 얻는다. 
# 이때 사용하는 활성화 함수는 일반적으로 다음과 같다.
# 
# - 은닉층의 활성화 함수: 음수를 제거하는 `relu` 함수
# - 이진 분류 모델의 최상위 층의 활성화 함수: 0과 1사이의 확률값을 계삲하는 `sigmoid` 함수

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp/master/slides/images/relu_sigmoid.png" style="width:700px;"></div>
# 
# 그림 출처: [Deep Learning with Python(Manning MEAP)](https://www.manning.com/books/deep-learning-with-python-second-edition)

# 몇 개의 층을 사용하는가와, 각 층마다 몇 개의 유닛(unit)을 사용하는가를
# 결정해야 하는데 이에 대해서 다음 장에서 자세히 설명한다.
# 여기서는 일단 다음과 같은 구성을 추천한다.
# 
# - 두 개의 연속된 밀집층
# - 각각 16개의 유닛
# 
# **참고**: 이진 분류 모델의 최상위 층은 스칼라 값을 출력하도록 하나의 유닛을 사용하는 
# `Dense` 밀집층을 사용한다. 
# 또한 활성화 함수로 0과 1사이의 확률값을 계삲하는 `sigmoid`를 활성화 함수로 사용한다.
# 그러면 [사이킷런의 로지스틱 회귀(logistic regression) 모델](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)처럼 작동한다.

# In[20]:


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


# **모델 컴파일링**
# 
# - 옵티마이저: `"rmsprop"`
#     - 일반적으로 추천되는 옵티마이저
# - 손실함수: `"binary_crossentropy"` (로그 손실)
#     - 확률 결과에 대한 오차 계산 용도로 최선임.
# - 평가지표: `"accuracy"`
#     - 분류 모델의 기본적인 평가지표

# In[21]:


model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])


# ### 모델 훈련 검증

# 훈련 중인 모델을 에포크마다 검증하려면 검증 세트를 따로 지정하면 된다.
# 여기서는 처음 10,000개의 샘플을 검증 세트로 활용한다.

# In[22]:


# 검증 세트
x_val = x_train[:10000]
y_val = y_train[:10000]

# 훈련셋
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]


# 훈련을 시작할 때 `validation_data` 옵션 인자를 지정하면 에포크마다 검증 세트를 이용하여 
# 훈련중인 모델을 평가한다.
# 
# - 에포크: 20
# - 배치 크기: 512
# - `validation_data=(x_val, y_val)`: 검증 세트 지정

# In[23]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# #### `History` 객체 활용

# `fit()` 메서드가 반환하는 객체는 `Callback` 클래스를 상속하는
# `History` 클래스의 인스턴이며, 케라스의 모든 모델 훈련과정 중에 발생하는 
# 다양한 정보를 저장한다.
# 
# **참고**: 콜백(`Callback`) 클래스에 대해서는 나중에 자세히 살펴볼 예정이다.

# - `params` 속성: 모델 훈련에 사용된 파라미터 저장

# In[24]:


history.params


# - `history` 속성: 평가지표를 사전 자료형으로 저장

# In[25]:


history_dict = history.history

history_dict.keys()


# 예를 들어, `history` 속성에 저장된 정보를 이용하여 
# 훈련셋와 검증 세트에 대한 에포크별 손실값과 정확도의 변화를 그래프로 그릴 수 있다.
# 
# * 훈련셋와 검증 세트에 대한 에포크별 손실값의 변화
#     - 훈련셋: 손실값 계속 감소
#     - 검증 세트: 다섯째 에포크 전후 정체 및 상승. 과대적합(overfitting) 발생.

# In[26]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")

plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# * 훈련셋와 검증 세트에 대한 에포크별 정확도의 변화
#     - 훈련셋: 정확도 계속 증가
#     - 검증 세트: 다섯째 에포크 전후 정체 및 감소. 과대적합(overfitting) 발생.

# In[27]:


plt.clf()    # 이전 이미지 삭제

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")

plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### 과대적합
# 
# **과대적합**(overfitting)은 모델이 훈련셋에 익숙해진다는 의미이다.
# 시험에 비유하면 연습문제의 답을 외워버리는 것을 의미한다.
# 과대적합을 방지하기 위한 다양한 기법은 다음 장(chapter)에서 다룬다.
# 위 문제의 경우 넷째 또는 다섯째 에포크 정도만 훈련 반복을 진행하면 된다.
# 아래 코드는 다시 처음부터 네 번의 에포크만을 사용하여 훈련한 결과를 보여준다.

# In[28]:


model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)


# 테스트셋에 대한 성능은 아래와 같이 88% 정도의 정확도를 보인다.
# 앞으로 보다 좋은 성능의 모델을 살펴볼 것이며, 현존하는 가장 좋은 모델의 정확도는 95% 정도이다.

# In[29]:


results = model.evaluate(x_test, y_test)
results


# ### 모델 활용
# 
# 훈련된 모델을 활용하려면 `predict()` 메서드를 이용한다.
# 
# - 0,99 이상 또는 0.01 이하의 경우: 매우 확실한 예측
# - 0.4 ~ 0.6: 불확실한 예측

# In[30]:


model.predict(x_test)


# 아래처럼 데이터셋이 클 경우 배치 단위로 묶어서 예측할 수도 있다.

# In[31]:


model.predict(x_test, batch_size=512)


# ### 연습문제

# 1. 두 개의 은닉층 대신 1 개 또는 3 개의 은닉층을 사용할 때 
#     검증 세트와 테스트셋에 대한 평가지표의 변화를 확인하라.
# 1. 각 은닉층에 사용된 유닛(unit)의 수를 8, 32, 64 등으로 변화시킨 후 
#     검증 세트와 테스트셋에 대한 평가지표의 변화를 확인하라.
# 1. `binary_crossentropy` 대신 `mse`를 손실함수로 지정한 후 
#     검증 세트와 테스트셋에 대한 평가지표의 변화를 확인하라.
# 1. `relu` 함수 대신 이전에 많이 사용됐었던 `tanh` 함수를 손실함수로 지정한 후 
#     검증 세트와 테스트셋에 대한 평가지표의 변화를 확인하라.

# ## 4.2 뉴스 기사 분류: 다중 클래스 분류

# ### 로이터(Reuter) 데이터셋

# - 1986년 로이터 통신사의 짧은 기사 모음
# - 케라스의 `reuters` 모듈의 `load_data()` 함수로 데이터 적재
#     - 훈련셋와 테스트셋로 분류됨.
# - 주제: 총 46개
# - 주제에 따른 기사 수 다름. 하지만 주제 당 최소 10개의 기사가 훈련셋에 포함됨.

# #### 데이터셋 적재
# 
# - `num_words=10000`: 10,000등 이내의 단어만 대상으로 함.
# - 데이터셋 크기: 11, 228
#     - 훈련셋 크기: 8,982
#     - 테스트셋 크기: 2,246

# In[32]:


from tensorflow.keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# In[33]:


len(train_data)


# In[34]:


len(test_data)


# 주제별 기사 수가 다르다.
# 훈련셋의 타깃에 사용된 값들의 빈도수를 확인하면 다음과 같다.

# In[35]:


from collections import Counter

target_counter = Counter(train_labels)
target_counter


# 가장 많이 언급된 주제는 총 3159번,
# 자장 적게 언급딘 주제는 총 10번 기사로 작성되었다.

# In[36]:


print(f"최대 기사 수: {max(target_counter.values())}")
print(f"최소 기사 수: {min(target_counter.values())}")


# 각 샘플은 정수들의 리스트이다.

# In[37]:


train_data[10]


# 각 샘플 리스트의 길이가 일반적으로 다르다.

# In[38]:


len(train_data[10])


# In[39]:


len(train_data[11])


# 각 샘플에 대한 레이블은 0부터 45까지의 정수로 표현된다.
# 예를 들어, 10번 기사의 주제는 3이다. 

# In[40]:


train_labels[10]


# 주제 3은 'earn'(이익)과 연관된다.
# 
# **참고**: 데이터 분석을 위해 반드시 필요한 사항은 아니지만 언급된 46개의 주제와 번호 사이의 관계는
# [GitHub: Where can I find topics of reuters dataset #12072](https://github.com/keras-team/keras/issues/12072)를 참조할 수 있다.

# In[41]:


reuter_topics = {'cocoa': 0,
                 'grain': 1,
                 'veg-oil': 2,
                 'earn': 3,
                 'acq': 4,
                 'wheat': 5,
                 'copper': 6,
                 'housing': 7,
                 'money-supply': 8,
                 'coffee': 9,
                 'sugar': 10,
                 'trade': 11,
                 'reserves': 12,
                 'ship': 13,
                 'cotton': 14,
                 'carcass': 15,
                 'crude': 16,
                 'nat-gas': 17,
                 'cpi': 18,
                 'money-fx': 19,
                 'interest': 20,
                 'gnp': 21,
                 'meal-feed': 22,
                 'alum': 23,
                 'oilseed': 24,
                 'gold': 25,
                 'tin': 26,
                 'strategic-metal': 27,
                 'livestock': 28,
                 'retail': 29,
                 'ipi': 30,
                 'iron-steel': 31,
                 'rubber': 32,
                 'heat': 33,
                 'jobs': 34,
                 'lei': 35,
                 'bop': 36,
                 'zinc': 37,
                 'orange': 38,
                 'pet-chem': 39,
                 'dlr': 40,
                 'gas': 41,
                 'silver': 42,
                 'wpi': 43,
                 'hog': 44,
                 'lead': 45}


# 실제로 10번 기사 내용을 확인해보면 'earn'과 관련되어 있어 보인다.
# 데이터를 해독(decoding)하는 방법은 IMDB 데이터셋의 경우와 동일하다.

# In[42]:


word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 10번 기사 내용은 다음과 같다.

# In[43]:


decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[10]])

decoded_newswire


# ### 데이터 전처리

# **데이터 벡터화**
# 
# IMDB의 경우와 동일하게 길이가 10,000인 벡터로 모든 샘플을 변환한다.

# In[44]:


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# ####  타깃의 원-핫-인코딩

# 앞서 보았듯이 타깃은 0부터 45 사이의 값이다.
# 이런 경우 정수를 텐서로 변환해서 사용하는 것보다
# **원-핫-인코딩**(one-hot-encoding) 기법을 적용하는 게 좋다.
# 
# 원-핫-인코딩은 멀티-핫-인코딩 기법과 유사하다.
# 차이점은 1인 오직 한 곳에서만 사용되고 나머지 항목은 모두 0이 된다.
# 예를 들어, 3은 길이가 46인 벡터(1차원 어레이)로 변환되는데
# 3번 인덱스에서만 1이고 나머지는 0이 된다.
# 
# 아래 함수는 원-핫-인코딩을 실행하는 함수이다.
# 입력값은 레이블 데이터셋 전체를 대상으로 함에 주의하라.

# In[45]:


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# 훈련셋의 레이블과 테스트셋의 레이블을 인코딩한다.

# In[46]:


y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)


# 인코딩된 레이블 하나를 살펴보자.

# In[47]:


y_train[0]


# #### `to_categorical()` 함수

# 원-핫-인코딩을 지원하는 함수를 케라스가 지원한다.
# 
# **참고**: 사용된 레이블의 최댓값에 1을 더한 값을 어레이의 길이로 사용한다.

# In[48]:


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# In[49]:


y_train[0]


# **참고**: 원-핫-인코딩, 멀티-핫-인코딩 등 정수를 사용하는 데이터를 범주형 데이터로 변환하는 
# 전처리 과정을 지원하는 층(layer)이 있다.
# 예를 들어 [tf.keras.layers.CategoryEncoding 층](https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding/)은 원-핫-인코딩과 멀티-핫-인코딩을 지원한다. 

# ### 모델 생성

# **모델 정의**
# 
# IMDB 데이터셋의 경우와는 달리 3 개 이상의 클래스로 분류하는 
# **다중 클래스 분류** 모델의 최종 층은 
# 분류해야 하는 클래스의 수 만큼의 유닛과 함께
# 각 클래스에 속할 확률을 계산하는 
# `softmax` 활성화 함수를 이용한다.
# 
# **참고**: 각 클래스에 속할 확률을 모두 더하면 1이 되며,
# 가장 높은 확률을 가진 클래스를 예측값으로 사용한다.
# 
# 반면에 이진 분류의 경우보다 복잡한 문제이기에 
# 은닉층의 유닛은 64개씩 정하여 보다 많은 정보를 상위 층으로 전달하도록 한다.

# In[50]:


model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])


# **모델 컴파일**
# 
# 다중 클래스 분류 모델의 손실함수는 `categorical_crossentropy`을 사용한다. 
# `categorical_crossentropy`는 클래스의 실제 분포와 예측 클래스의 분포 사이의 
# 오차를 측정하는 손실함수이며, 
# 자세한 내용은 생략한다.

# In[51]:


model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# ### 모델 훈련 및 검증

# **검증 세트 지정**
# 
# 처음 1,000개의 샘플을 검증 세트 용도로 사용한다.

# In[52]:


# 검증 세트
x_val = x_train[:1000]
y_val = y_train[:1000]

# 훈련셋
partial_x_train = x_train[1000:]
partial_y_train = y_train[1000:]


# **모델 훈련**
# 
# 훈련 방식은 이전과 동일하다.

# In[53]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# **손실값의 변화**

# * 훈련셋와 검증 세트에 대한 에포크별 손실값의 변화
#     - 훈련셋: 손실값 계속 감소
#     - 검증 세트: 아홉번 째 에포크 전후 정체 및 상승. 과대적합(overfitting) 발생.

# In[54]:


loss = history.history["loss"]

val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# **정확도의 변화**

# * 훈련셋와 검증 세트에 대한 에포크별 정확도의 변화
#     - 훈련셋: 정확도 계속 증가
#     - 검증 세트: 아홉번 째 에포크 전후 정체 및 감소. 과대적합(overfitting) 발생.

# In[55]:


plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# **모델 재훈련**
# 
# 9번 에포크를 지나면서 과대적합이 발생하는 것으로 보인다. 
# 따라서 에포크 수를 9로 줄이고 처음부터 다시 훈련시켜보자.
# 모델 구성부터, 컴파일, 훈련을 모두 다시 시작해야 
# 가중치와 편향이 초기화된 상태서 훈련을 시작한다.

# In[56]:


model = keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)


# 훈련된 모델을 이용한 테스트셋에 대한 예측의 정확도는 80% 정도이다.

# In[57]:


results = model.evaluate(x_test, y_test)

results


# 80%의 정확도가 얼마나 좋은지/나쁜지를 판단하려면 무작위로 찍을 때의 정확도를 계산해봐야 한다.
# 아래 코드가 이를 실천하며, 20% 정도의 정확도가 나온다.
# 따라서 80% 정도의 정확도는 상당히 좋은 편이다.

# In[58]:


import copy

# 원 데이터를 건드리지 않기 위해 사본 사용
test_labels_copy = copy.copy(test_labels)

# 무작위로 섞은 후 원 데이터의 순서와 비교
np.random.shuffle(test_labels_copy)
hits_array = test_labels == test_labels_copy

# 1 또는 0으로만 이루어졌기에 평균값을 계산하면 무작위 선택의 정확도를 계산함
hits_array.mean()


# ### 예측하기

# 훈련된 모델을 테스트셋에 적용한다.

# In[59]:


predictions = model.predict(x_test)


# 예측값의 모두 길이가 46인 1차원 어레이다.

# In[60]:


predictions[0].shape


# 예측값은 46개 클래스에 들어갈 확률들로 이루어지며 합은 1이다.

# In[61]:


np.sum(predictions[0])


# 가장 큰 확률값을 가진 인덱스가 모델이 예측하는 클래스가 된다.
# 예를 들어 테스트셋의 0번 샘플(로이터 기사)은 3번 레이블을 갖는다고 예측된다.

# In[62]:


np.argmax(predictions[0])


# ### 정수 레이블 사용법

# 정수 텐서 레이블(타깃)을 이용하여 훈련하려면 모델을 컴파일할 때 손실함수로 
# `sparse_categorical_crossentropy`를 
# 사용하면 된다.
# 
# ```python
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
# 
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# ```

# ### 은닉층에 사용되는 유닛 개수

# 은닉층에 사용되는 유닛은 마지막 층의 유닛보다 많아야 한다.
# 그렇지 않으면 정보전달 과정에 병목현상(bottleneck)이 발생할 수 있다.
# 아래 코드의 둘째 은닉층은 4 개의 유닛만을 사용하는데 
# 훈련된 모델의 성능이 많이 저하된다.

# In[63]:


model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))


# 테스트셋에 대한 정확도가 80% 정도에서 65% 정도로 낮아진다.

# In[64]:


model.evaluate(x_test, y_test)


# ### 연습문제
# 
# 1. 은닉층의 유닛 개수를 32, 128 등 여러 값으로 실험해 보아라.
# 1. 은닉층의 수를 1개 또는 3개로 바꿔 보아라.

# ## 4.3 주택가격 예측: 회귀

# 이진 분류, 다중 클래스 분류 모델은 지정된 숫자들로 이루어진 특정 클래스의 번호 하나를 예측한다.
# 반면에 임의의 수를 예측하는 문제는 **회귀**(regression)이라 부른다. 
# 예를 들어 온도 예측, 가격 예측 등을 다루는 것이 회귀 문제이다. 
# 여기서는 보스턴 시의 주택가격을 예측하는 회귀 문제를 예제로 다룬다.
# 
# **주의사항**: '로지스틱 회귀'(logistic regression) 알고리즘는 분류 모델임에 주의하라.

# ### 보스턴 주택가격 데이터셋

# 사용하는 데이터셋은 다음과 같다.
# 
# - 1970년대 중반의 미국 보스턴 시내와 외곽의 총 506개 지역별 중간 주택가격.
#     즉, 매우 적은 수의 데이터셋임.
# - 케라스의 `boston_housing` 모듈의 `load_data()` 함수로 데이터 적재
#     - 훈련셋와 테스트셋로 분류됨.
# - 지역별 샘플
#     - 특성: 총 13 개. 지역별 범죄율, 토지 비율, 재산세율, 학생 대 교사 비율 등.
#     - 타깃: 주택가격
# - 참고: [위키독스: 보스턴 주택가격 데이터셋 소개](https://wikidocs.net/49966)

# **보스턴 주택가격 데이터셋 적재**
# 
# - 데이터셋 크기: 506
#     - 훈련셋: 404
#     - 테스트셋: 102
# - 샘플 특성 수: 13

# In[65]:


from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# In[66]:


train_data.shape


# In[67]:


test_data.shape


# 훈련셋 샘플의 타깃은 아래처럼 범위가 지정되지 않은 부동소수점 값이다. 

# In[68]:


train_targets[:10]


# ### 데이터 전처리

# 특성에 따라 사용되는 값들의 크기가 다르다. 
# 어떤 특성은 0과 1사이, 다른 특성은 한 자리리부터 세 자리의 수를 갖기도 한다.

# In[69]:


import pandas as df

df.DataFrame(train_data).describe()


# **데이터 정규화**
# 
# 따라서 모든 특성의 값을 **정규화** 해주어야 모델 훈련이 더 잘된다.
# 모든 특성값들을 특성별로 표준 정규분포를 따르도록 한다. 
# 즉, 평균값 0, 표준편차 1이 되도록 특성값을 특성별로 변환한다.
# 
# **주의사항**: 테스트셋의 정규화는 훈련셋의 평균값과 표준편차를 이용해야 한다.
# 이유는 테스트셋의 정보는 모델 훈련에 절대로 사용되지 않아야 하기 때문이다. 

# In[70]:


# 훈련셋의 평균값
mean = train_data.mean(axis=0)

# 훈련셋 정규화
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 테스트셋 정규화: 훈련셋의 평균값과 표준편차 활용
test_data -= mean
test_data /= std


# ### 모델 구현

# **모델 정의**

# 이전과는 달리 모델 구성과 컴파일을 동시에 진행하는 함수를 이용한다.
# 
# - 은닉층: 데이터셋이 작으므로 두 개만 사용.
# - 각 은닉층의 유닛 수: 인자로 받도록 함. 아래 예제에서는 64 사용.
# - 마지막 층: 활성화 함수 없음. 회귀 모델이기 때문임.
# - 손실함수: **평균제곱오차**(mse). 회귀 모델의 일반적인 손실함수
# - 평가지표: **평균절대오차**(mae, mean absolute error)
# 
# **참고**: 데이터셋이 클 수록 보다 많은 층과 보다 많은 유닛 사용하는 것이 일반적임.

# In[71]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# ### K-겹 교차검증 활용

# 데이터셋이 작기에 훈련 중에 사용할 검증 세트를 따로 분리하는 것은 훈련의 효율성을 떨어뜨린다.
# 대신에 **K-겹 교차검증**을(K-fold cross-validation) 사용한다.
# 아래 이미지는 3-겹 교차검증을 사용할 때 훈련 중에 사용되는 훈련셋과 검증셋의 사용법을 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/3-fold-cross-validation.png" style="width:600px;"></div>
# 
# 그림 출처: [Deep Learning with Python(Manning MEAP)](https://www.manning.com/books/deep-learning-with-python-second-edition)

# **예제: 4-겹 교차검증**
# 
# - 에포크 수: 500
# - `validation_data` 옵션 인자 활용
#     - 교차검증과 에포크마다 평가지표 저장됨.
# - `verbose=0`: 손실값과 평가지표를 출력하지 않음.

# In[72]:


k = 4
num_val_samples = len(train_data) // k

num_epochs = 500
all_mae_histories = []   # 모든 에포크에 대한 평균절대오차 저장

for i in range(k):       # 교차 검증
    
    print(f"{i+1}번 째 폴드(fold) 훈련 시작")

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    
    model = build_model()    # 유닛 수: 64
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)


# **K-겹 교차검증 훈련 과정 그래프: 평가지표 기준**
# 
# 500번의 에포크마다 4 번의 교차 검증을 진행하였기에
# 에포크 별로 검증세트를 대상으로하는 평균절대오차의 평균값을 계산한다.

# In[73]:


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# 에포크별 평균절대오차의 평균값의 변화를 그래프로 그리면 다음과 같다.

# In[74]:


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


# 첫 10개의 에포크의 성능이 매우 나쁘기에 그 부분을 제외하고 그래프를 그리면 보다 정확하게 
# 변환 과정을 파악할 수 있다.

# In[75]:


truncated_mae_history = average_mae_history[10:]

plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


# **재훈련**
# 
# - 130번 째 에포크를 전후로 과대적합 발생함.
# - 130번의 에포크만 사용해서 모델 재훈련

# In[76]:


model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)


# 재훈련된 모델의 테스트셋에 대한 성능을 평가하면 
# 주택가격 예측에 있어서 평균적으로 2,500달러 정도의 차이를 갖는다.

# In[77]:


test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score


# ### 모델 활용
# 
# - 새로운 데이터에 대한 예측은 `predict()` 메서드를 활용한다. 

# In[78]:


predictions = model.predict(test_data)
predictions[0]


# ### 연습문제

# 사이킷런의 `KFold`를 이용하면 봅다 간단하게 K-겹 교차검증을 진행할 수 있다.

# In[79]:


from sklearn.model_selection import KFold

k = 4
num_epochs = 500

kf = KFold(n_splits=k)
all_mae_histories = []

for train_index, val_index in kf.split(train_data, train_targets):
    
    val_data, val_targets = train_data[val_index], train_targets[val_index]
    partial_train_data, partial_train_targets = train_data[train_index], train_targets[train_index]
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)

    mae_history = history.history["val_mae"]    
    all_mae_histories.append(mae_history)


# In[80]:


test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score

