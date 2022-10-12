#!/usr/bin/env python
# coding: utf-8

# (ch:working_with_keras)=
# # 케라스 모델 고급 활용법

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 케라스 모델 활용법](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-working_with_keras.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**
# 
# - 모델 구성법
# - 모델 훈련 모니터링
# - 사용자 정의 모델 훈련 및 평가

# ## 케라스의 다양한 활용법

# 케라스를 이용하여 매우 간단한 방식부터 매우 복잡한 방식까지 다양한 방식으로 
# 필요한 수준의 모델을 구성할 수 있다.
# 또한 케라스의 모델과 층은 모두 각각 `Model` 클래스와 `Layer` 클래스를 상속하기에 
# 다른 모델에서 사용된 요소들을 재활용하기에도 용이하다.
# 
# 여기서는 주어진 문제에 따른 케라스 모델 구성법과 훈련법의 다양한 방식을 살펴본다. 

# ## 케라스 모델 구성법

# 케라스를 이용하여 세 가지 방식으로 딥러닝 모델을 구성할 수 있다.
# 
# - `Sequential` 모델 활용: 층으로 스택을 쌓아 만든 모델
# - 함수형 API 활용: 가장 많이 사용됨.
# - 모델 서브클래싱: 모든 것을 사용자가 지정.

# ### 모델 구성법 1: `Sequential` 모델

# 층으로 스택을 쌓아 만든 모델이며 가장 단순하다.
# 
# - 하나의 입력값과 하나의 출력값만 사용 가능
# - 층을 지정된 순서대로 적용

# **`Sequential` 클래스**

# ```python
# from tensorflow import keras
# from tensorflow.keras import layers
# 
# model = keras.Sequential([
#     layers.Dense(64, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# ```

# 층의 추가는 `add` 메서드를 이용할 수도 있다.
# 더해진 순서대로 층이 쌓인다.
# 예를 들어, 아래 코드는 앞서 정의한 모델과 동일한 모델을 구성한다.

# ```python
# model = keras.Sequential()
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))
# ```

# **모델의 가중치와 `build()` 메서드**

# 지금 당장 모델의 가중치를 확인하려 하면 오류가 발생한다.

# ```python
# >>> model.weights
# ...
# ValueError: Weights for model sequential_1 have not yet been created. 
# Weights are created when the Model is first called on inputs or 
# `build()` is called with an `input_shape`.
# ```

# 이유는 입력값이 들어와야 입력 텐서의 모양<font size='2'>shape</font>을 보고 
# 가중치와 편향 텐서의 모양을 정할 수 있기 때문이다.
# 이 과정은 모델 훈련이 시작될 때 제일 먼저 호출되는 
# `build()` 메서드가 담당한다. 
# 아래 코드 샘플은 {numref}`%s장 <ch:keras-tf>`에서
# `SimpleDense`를 선언할 때 사용된 `build()` 메서드를 보여준다.

# ```python
# def build(self, input_shape):
#     input_dim = input_shape[-1]   # 입력 샘플의 특성 수
#     self.W = self.add_weight(shape=(input_dim, self.units),
#                              initializer="random_normal")
#     self.b = self.add_weight(shape=(self.units,),
#                              initializer="zeros")
# ```

# **`build()` 메서드 호출**

# `build()` 메서드에 입력 샘플의 차원, 즉 특성의 개수에 대한 정보를 제공하여 호출하면
# 가중치 텐서가 무작위로 초기화된 형식으로 생성된다.
# 즉, **모델 빌드**가 완성된다.
# 
# - `input_shape` 키워드 인자: `(None, 특성수)`
# - `None`은 배치의 크기는 상관하지 않는다는 것을 의미한다.
#     실제로 가중치와 편향 텐서의 모양은 입력 샘플의 차원과 각 층에서 사용되는 유닛(뉴런)의 개수에만 의존한다.

# ```python
# >>> model.build(input_shape=(None, 3))
# ````

# **층별 가중치 텐서**

# 모델 빌드가 완성되면 `weights` 속성에 생성된 모델 훈련에 필요한 모든 가중치와 편향이 저장된다.
# 위 모델에 대해서 층별로 가중치와 편향 텐서 하나씩 총 4 개의 텐서가 생성된다.

# ```python
# >>> len(model.weights)
# 4
# ```

# **`summary()` 메서드**

# 완성된 모델의 요악한 내용은 확인할 수 있다.
# 
# - 모델과 층의 이름
# - 층별 파라미터 수
# - 파라미터 수

# ```python
# >>> model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_2 (Dense)              (None, 64)                256       
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 906
# Trainable params: 906
# Non-trainable params: 0
# _________________________________________________________________
# ```

# **`name` 인자**
# 
# 모델 또는 층을 지정할 때 생성자 메서등의 `name` 키워드 인자를 이용하여 이름을 지정할 수도 있다.

# ```python
# model = keras.Sequential(name="my_example_model")
# model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
# model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
# ```

# `summary()` 결과에 모델과 층의 이름이 추가된다.

# ```python
# >>> model.build((None, 3))
# >>> model.summary()
# Model: "my_example_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# my_first_layer (Dense)       (None, 64)                256       
# _________________________________________________________________
# my_last_layer (Dense)        (None, 10)                650       
# =================================================================
# Total params: 906
# Trainable params: 906
# Non-trainable params: 0
# ```
# 

# **`Input()` 함수**

# 모델 구성하는 중간에 그때까지의 모델 구성을 확인하기 위해
# `Input()`함수를 이용할 수 있다.
# `Input()` 함수는 모델 훈련에 사용되는 훈련 샘플의 모양 정보를 제공하는
# 가상의 텐서인 `KerasTensor` 객체를 생성한다.

# ```python
# model = keras.Sequential()
# model.add(keras.Input(shape=(3,)))
# model.add(layers.Dense(64, activation="relu"))
# ```

# `Input()` 함수의 `shape` 키워드 인자에 사용되는 값은 각 샘플의 특성 수이며,
# 앞서 `build()` 메서드의 인자와 다른 형식으로 사용됨에 주의해야 한다.

# 이제 층을 추가할 때마다 `build()` 메서드를 실행할 필요 없이 바로 `summary()`를 실행할 수 있다.

# ```python
# >>> model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_4 (Dense)              (None, 64)                256       
# =================================================================
# Total params: 256
# Trainable params: 256
# Non-trainable params: 0
# _________________________________________________________________
# ```

# ```python
# >>> model.add(layers.Dense(10, activation="softmax"))
# >>> model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_4 (Dense)              (None, 64)                256       
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 906
# Trainable params: 906
# Non-trainable params: 0
# _________________________________________________________________
# ```

# ### 모델 구성법 2: 함수형 API

# `Sequential` 클래스를 사용하면 하나의 입력과 하나의 출력만을 지원한다.
# 반면에 함수형 API를 이용하면 다중 입력과 다중 출력을 지원하는 모델을 구성할 수 있다.
# 가장 많이 사용되는 모델 구성법이며, 사용법이 매우 간단하다.
# 사용법은 간단하다.

# 
# ```python
# Model(inputs, outputs)
# ```
# 
# - `Model`: 케라사의 기본 모델 클래스
# - `inputs` 인자: 한 개 이상의 케라스텐서(`KerasTensor`) 객체 이루어진 리스트
# - `outputs` 인자: 한 개 이사의 출력층으로 이루어진 리스트

# **기본 활용법**

# 앞서 살펴 본 `Sequential` 모델을 함수형 API를 이용하여 구성하면 다음과 같다.

# ```python
# inputs = keras.Input(shape=(3,), name="my_input")          # 입력층
# features = layers.Dense(64, activation="relu")(inputs)     # 은닉층
# outputs = layers.Dense(10, activation="softmax")(features) # 출력층
# model = keras.Model(inputs=inputs, outputs=outputs)        # 모델 지정
# ```

# 사용된 단계들을 하나씩 살펴보자.

# 입력층

# ```python
# >>> inputs = keras.Input(shape=(3,), name="my_input")
# ```

# 생성된 값은 `KerasTensor`이다.

# ```python
# >>> type(inputs)
# keras.engine.keras_tensor.KerasTensor
# ```

# 케라스텐서(`KerasTensor`)의 모양에서 `None`은 배치 사이즈, 즉 
# 하나의 훈련 스텝에 사용되는 샘플의 수를 대상으로 하며, 
# 임의의 크기의 배치를 처리할 수 있다는 의미로 사용된다.

# 텐서 항목의 자료형은 `float32`가 기본값으로 사용된다.

# ```python
# >>> inputs.dtype
# float32
# ```

# 은닉층

# ```python
# >>> features = layers.Dense(64, activation="relu")(inputs)
# ```

# 은닉층의 결과는 항상 `KerasTensor`이다.

# ```python
# >>> type(features)
# keras.engine.keras_tensor.KerasTensor
# ```

# 출력값의 모양은 유닛의 개수에 의존한다.

# ```python
# >>> features.shape
# TensorShape([None, 64])
# ```

# 출력층

# ```python
# >>> outputs = layers.Dense(10, activation="softmax")(features)
# ```

# 출력층의 결과도 `KerasTensor`이다.

# ```python
# >>> type(outputs)
# keras.engine.keras_tensor.KerasTensor
# ```

# 모델 빌드

# 입력층과 출력층을 이용하여 모델을 지정한다.

# ```python
# >>> model = keras.Model(inputs=inputs, outputs=outputs)
# ```

# `summary()` 의 실행결과는 이전과 동일하다.

# ```python
# >>> model.summary()
# Model: "functional_1" 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param # 
# =================================================================
# my_input (InputLayer)        [(None, 3)]               0 
# _________________________________________________________________
# dense_6 (Dense)              (None, 64)                256 
# _________________________________________________________________
# dense_7 (Dense)              (None, 10)                650 
# =================================================================
# Total params: 906 
# Trainable params: 906 
# Non-trainable params: 0 
# _________________________________________________________________
# ```

# `KerasTensor`의 역할

# 앞서 보았듯이 케라스텐서는 모델 훈련에 사용되는 텐서의 모양에 대한 정보를 제공하는 
# **가상의 텐서**이다.
# 빌드되는 모델은 입력 케라스텐서부터 출력 케라스텐서까지 각 층에 저장된 
# 텐서의 모양 정보를 이용하여 가중치 텐서와 편향 텐서를 생성하고 초기화한다.

# **다중 입력, 다중 출력 모델 구성법**

# 다중 입력과 다중 출력을 지원하는 모델을 구성하는 방법을 예제를 이용하여 설명한다.

# 예제: 고객 요구사항 접수 모델

# 고객의 요구사항의 처리할 때 필요한 우선순위와 담당부서를 지정하는 시스템을 구현하려 한다.
# 시스템에 사용될 딥러닝 
# 모델은 세 개의 입력과 두 개의 출력을 사용한다. 
# 
# - 입력 사항 세 종류
#     - `title`: 요구사항 문자열을 0과 1로 구성된 텐서로 변환한 텐서 입력값. 최대 10,000 종류의 단어 사용.
#     - `text_body`: 요구사항 문자열을 0과 1로 구성된 텐서로 변환한 텐서 입력값. 최대 10,000 종류의 단어 사용.
#     - `tags`: 사용자에 의한 추가 선택 사항 100개 중 여러 개 선택. 멀티-핫 인코딩된 텐서 입력값.
# - 출력 사항 두 종류
#     - `priority`: 요구사항 처리 우선순위. 0에서 1사이의 값. `sigmoid` 활성화 함수 활용.
#     - `department`: 네 개의 요구사항 처리 담당 부서 중 하나 선택. `softmax` 활성화 함수 활용.

# 함수형 API를 이용하여 모델을 구현하면 다음과 같다.

# ```python
# vocabulary_size = 10000    # 요구사항에 사용되는 단어 총 수
# num_tags = 100             # 태그 수
# num_departments = 4        # 부서 수
# 
# # 입력층: 세 종류
# title = keras.Input(shape=(vocabulary_size,), name="title")
# text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
# tags = keras.Input(shape=(num_tags,), name="tags")
# 
# # 은닉층
# features = layers.Concatenate()([title, text_body, tags]) # shape=(None, 10000+10000+100)
# features = layers.Dense(64, activation="relu")(features)
# 
# # 출력층: 두 종류
# priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
# department = layers.Dense(
#     num_departments, activation="softmax", name="department")(features)
# 
# # 모델 빌드: 입력값과 출력값 모두 리스트 사용
# model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
# ```

# 모델 컴파일 과정에서 지정된 타깃 개수만큼의 손실함수와 측정 기준을 지정해야 한다.
# 
# - 손실함수(loss)
#     - `priority` 대상: `mean_squared_error`
#     - `department` 대상: `categorical_crossentropy`
# 
# - 평가지표(metrics): 평가지표는 여러 개를 사용할 수 있기에 대상 별로 리스트로 지정함.
#     - `priority` 대상: `["mean_absolute_error"]`
#     - `department` 대상: `["accuracy"]`

# ```python
# model.compile(optimizer="adam",
#               loss=["mean_squared_error", "categorical_crossentropy"],
#               metrics=[["mean_absolute_error"], ["accuracy"]])
# ```

# 모델 훈련은 `fit()` 함수에 세 개의 훈련 텐서로 이루어진 리스트와 
# 두 개의 타깃 텐서로 이루어진 리스트를 지정한 후에 실행한다. 

# ```python
# model.fit([title_data, text_body_data, tags_data],
#           [priority_data, department_data],
#           ...
#           )
# ```

# 모델 평가도 훈련과 동일한 방식의 인자가 사용된다.

# ```python
# model.evaluate([title_data, text_body_data, tags_data],
#                [priority_data, department_data])
# ```

# 예측값은 두 개의 어레이로 구성된 리스트이다.

# ```python
# priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])
# ```

# **사전 객체 활용**
# 
# 입력층과 출력층의 이름을 이용하여 사전 형식으로 입력값과 출력값을 지정할 수 있다.

# ```python
# model.compile(optimizer="adam",
#               loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
#               metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]})
# 
# model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
#           {"priority": priority_data, "department": department_data},
#           epochs=1)
# 
# model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
#                {"priority": priority_data, "department": department_data})
# 
# priority_preds, department_preds = model.predict(
#     {"title": title_data, "text_body": text_body_data, "tags": tags_data})
# ```

# **층 연결 구조**

# `plot_model()`을 이용하여 층 연결 구조를 그래프로 나타낼 수 있다.
# 
# ```python
# >>> keras.utils.plot_model(model, "ticket_classifier.png")
# ```

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ticket_classifier.png" style="width:400px;"></div>

# **주의사항**: `pydot` 파이썬 모듈과 graphviz 라는 프로그램이 컴퓨터에 설치되어 있어야 한다.
# 
# - `pydot` 모듈 설치: `pip install pydot`
# - graphviz 프로그램 설치: [https://graphviz.gitlab.io/download/](https://graphviz.gitlab.io/download/)
# - 구글 코랩에서 기본으로 지원됨.

# 입력 텐서와 출력 텐서의 모양을 함께 표기할 수도 있다.
# 
# ```python
# >>> keras.utils.plot_model(model, "ticket_classifier_with_shape_info.png", show_shapes=True)
# ```

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/ticket_classifier_with_shapes.png" style="width:900px;"></div>

# #### 모델 재활용

# 훈련된 모델의 특성을 이용하여 새로운 모델을 빌드할 수 있다.
# 먼저 모델의 `layers` 속성을 이용하여 사용된 층에 대한 정보를 확인한다. 
# `layers` 속성은 사용된 층들의 객체로 이루어진 리스트를 가리킨다.

# ```python
# >>> model.layers
# [<tensorflow.python.keras.engine.input_layer.InputLayer at 0x7fa963f9d358>,
#  <tensorflow.python.keras.engine.input_layer.InputLayer at 0x7fa963f9d2e8>,
#  <tensorflow.python.keras.engine.input_layer.InputLayer at 0x7fa963f9d470>,
#  <tensorflow.python.keras.layers.merge.Concatenate at 0x7fa963f9d860>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7fa964074390>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7fa963f9d898>,
#  <tensorflow.python.keras.layers.core.Dense at 0x7fa963f95470>]
# >>> model.layers[3].input
# [<tf.Tensor "title:0" shape=(None, 10000) dtype=float32>,
#  <tf.Tensor "text_body:0" shape=(None, 10000) dtype=float32>,
#  <tf.Tensor "tags:0" shape=(None, 100) dtype=float32>]
# >>> model.layers[3].output
# <tf.Tensor "concatenate/concat:0" shape=(None, 20100) dtype=float32>
# ```

# 예를 들어, 3번 인덱스에 해당하는 층의 입력값과 출력값에 대한 정보는 아래처럼 확인할 수 있다.

# ```python
# >>> model.layers[3].input
# [<tf.Tensor "title:0" shape=(None, 10000) dtype=float32>,
#  <tf.Tensor "text_body:0" shape=(None, 10000) dtype=float32>,
#  <tf.Tensor "tags:0" shape=(None, 100) dtype=float32>]
# ```

# ```python
# >>> model.layers[3].output
# <tf.Tensor "concatenate/concat:0" shape=(None, 20100) dtype=float32>
# ```

# 출력층을 제외한 나머지 층을 재활용해보자.
# 출력층은 5번과 6번 인덱스에 위치하기에 4번 인덱스가
# 가리키는 (은닉)층의 출력 정보를 따로 떼어낸다.

# ```python
# >>> features = model.layers[4].output
# ```

# 이제 출력층에 문제해결의 어려움 정도를 "quick", "medium", "difficult"로
# 구분하는 어려움(difficulty) 정도를 판별하는 층을 추가해보자.
# 먼저, `difficulty` 층을 준비한다.

# ```python
# >>> difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
# ```

# 준비된 `'difficulty'` 층을 출력층으로 추가하여 
# `priority`, `department`, `difficulty`
# 세 개의 출력값을 생성하는 새로운 모델을 구성한다.

# ```python
# new_model = keras.Model(
#     inputs=[title, text_body, tags],
#     outputs=[priority, department, difficulty])
# ```

# 새로 생성된 모델은 기존에 훈련된 모델의 가중치,
# 즉, 은닉층에 사용된 가중치는 그대로 사용되며,
# 모델 구성 그래프는 다음과 같다.
# 
# ```python
# >>> keras.utils.plot_model(new_model, "updated_ticket_classifier.png", show_shapes=True)
# ```
# 
# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/updated_ticket_classifier.png" style="width:900px;"></div>

# 요약 결과는 다음과 같다.

# ```python
# >>> new_model.summary()
# Model: "model_2"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# title (InputLayer)              [(None, 10000)]      0                                            
# __________________________________________________________________________________________________
# text_body (InputLayer)          [(None, 10000)]      0                                            
# __________________________________________________________________________________________________
# tags (InputLayer)               [(None, 100)]        0                                            
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 20100)        0           title[0][0]                      
#                                                                  text_body[0][0]                  
#                                                                  tags[0][0]                       
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 64)           1286464     concatenate[0][0]                
# __________________________________________________________________________________________________
# priority (Dense)                (None, 1)            65          dense_8[0][0]                    
# __________________________________________________________________________________________________
# department (Dense)              (None, 4)            260         dense_8[0][0]                    
# __________________________________________________________________________________________________
# difficulty (Dense)              (None, 3)            195         dense_8[0][0]                    
# ==================================================================================================
# Total params: 1,286,984
# Trainable params: 1,286,984
# Non-trainable params: 0
# __________________________________________________________________________________________________
# ```

# ### 모델 구성법 3: 서브클래싱

# 케라스 모델과 호환되는 모델 클래스를 직접 선언하여 활용하려면 `keras.Model` 클래스를 상속해야 한다.
# 이런 방식을 **서브클래싱**<font size='2'>subclassing</font>이라 부른다.
# 
# 서브클래싱은 `keras.Model` 클래스를 상속하면서 기본적으로 아래 두 메서드를 목적에 맞추어 
# 재정의<font size='2'>overriding</font>하는 방식으로 진행된다.
# 
# - `__init__()` 메서드(생성자): 은닉층과 출력층의 구성요소 지정.
# - `call()` 메서드: 입력으로부터 출력을 만들어내는 순전파 과정 묘사.
# 
# 앞서 함수형 API로 구성한 고객 요구사항 접수 모델을 서브클래싱을 기법을 이용하여 구현하면 다음과 같다.
# 
# **참고**: `keras.layers.Layer`를 상속하여 사용자 정의 층을 선언하는 방식과 거의 유사하다([3장 6절](https://codingalzi.github.io/dlp/notebooks/dlp03_introduction_to_keras_and_tf.html) 참조).

# ```python
# class CustomerTicketModel(keras.Model):
# 
#     def __init__(self, num_departments):
#         super().__init__()
#         self.concat_layer = layers.Concatenate()
#         self.mixing_layer = layers.Dense(64, activation="relu")
#         self.priority_scorer = layers.Dense(1, activation="sigmoid")
#         self.department_classifier = layers.Dense(
#             num_departments, activation="softmax")
# 
#     def call(self, inputs):               # inputs: 사전 객체 입력값. 모양은 미정.
#         title = inputs["title"]
#         text_body = inputs["text_body"]
#         tags = inputs["tags"]
# 
#         features = self.concat_layer([title, text_body, tags])    # 은닉층
#         features = self.mixing_layer(features)
#         priority = self.priority_scorer(features)                 # 출력층
#         department = self.department_classifier(features)
#         return priority, department                               # outputs
# ```

# 모델 구성은 해당 모델의 객체를 생성하면 된다.
# 다만 `Layer`의 경우처럼 가중치는 실제 데이터와 함께 호출되지 전까지 생성되지 않는다.

# ```python
# >>> model = CustomerTicketModel(num_departments=4)
# ```

# 컴파일, 훈련, 평가, 예측은 이전과 완전히 동일한 방식으로 실행된다.

# **서브클래싱 기법의 장단점**

# - 장점
#     - `call()` 함수를 이용하여 층을 임의로 구성할 수 있다.
#     - `for` 반복문 등 파이썬 프로그래밍 모든 기법을 적용할 수 있다.
# - 단점
#     - 모델 구성을 전적으로 책임져야 한다.
#     - 모델 구성 정보가 `call()` 함수 외부로 노출되지 않아서
#         앞서 보았던 그래프 표현을 사용할 수 없다. 

# ### 모델 구성법 혼합

# 소개된 세 가지 방식을 임의로 혼합하여 활용할 수 있다. 

# **예제: 서브클래싱 모델을 함수형 모델에 활용하기** (강추!!!)

# ```python
# class Classifier(keras.Model):
# 
#     def __init__(self, num_classes=2):
#         super().__init__()
#         if num_classes == 2:
#             num_units = 1
#             activation = "sigmoid"
#         else:
#             num_units = num_classes
#             activation = "softmax"
#         self.dense = layers.Dense(num_units, activation=activation)
# 
#     def call(self, inputs):
#         return self.dense(inputs)
# 
# inputs = keras.Input(shape=(3,))
# features = layers.Dense(64, activation="relu")(inputs)
# outputs = Classifier(num_classes=10)(features)
# model = keras.Model(inputs=inputs, outputs=outputs)
# ```

# **예제: 함수형 모델을 서브클래싱 모델에 활용하기**

# ```python
# inputs = keras.Input(shape=(64,))
# outputs = layers.Dense(1, activation="sigmoid")(inputs)
# binary_classifier = keras.Model(inputs=inputs, outputs=outputs)
# 
# class MyModel(keras.Model):
# 
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.dense = layers.Dense(64, activation="relu")
#         self.classifier = binary_classifier
# 
#     def call(self, inputs):
#         features = self.dense(inputs)
#         return self.classifier(features)
# 
# model = MyModel()
# ```

# ## 훈련 모니터링

# 케라스 모델의 구성, 훈련, 평가, 예측은 정해진 방식으로 차례대로 이루어진다.
# 아래 코드는 MNIST 데이터셋을 이용한 모델 훈련 전반 과정을 보여준다.

# ```python
# from tensorflow.keras.datasets import mnist
# 
# def get_mnist_model():
#     inputs = keras.Input(shape=(28 * 28,))
#     features = layers.Dense(512, activation="relu")(inputs)
#     features = layers.Dropout(0.5)(features)
#     outputs = layers.Dense(10, activation="softmax")(features)
#     model = keras.Model(inputs, outputs)
#     return model
# 
# (images, labels), (test_images, test_labels) = mnist.load_data()
# images = images.reshape((60000, 28 * 28)).astype("float32") / 255
# test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
# train_images, val_images = images[10000:], images[:10000]
# train_labels, val_labels = labels[10000:], labels[:10000]
# 
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=3,
#           validation_data=(val_images, val_labels))
# test_metrics = model.evaluate(test_images, test_labels)
# predictions = model.predict(test_images)
# ```

# ### 사용자 정의 평가지표(`metrics`) 활용

# **`Metric` 클래스 상속**
# 
# 아래 세 개의 메서드를 재정의(overriding)해야 한다.
# 
# - `update_state()`
# - `result()`
# - `reset_state()`
# 
# 아래 코드는 평균제곱근오차(RMSE)를 평가지표로 사용하는 클래스를 
# 이용하는 모델 훈련을 소개한다.

# ```python
# import tensorflow as tf
# 
# class RootMeanSquaredError(keras.metrics.Metric):
# 
#     def __init__(self, name="rmse", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
#         self.total_samples = self.add_weight(
#             name="total_samples", initializer="zeros", dtype="int32")
# 
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
#         mse = tf.reduce_sum(tf.square(y_true - y_pred))
#         self.mse_sum.assign_add(mse)
#         num_samples = tf.shape(y_pred)[0]
#         self.total_samples.assign_add(num_samples)
# 
#     def result(self):
#         return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))
# 
#     def reset_state(self):
#         self.mse_sum.assign(0.)
#         self.total_samples.assign(0)
# ```

# ```python
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy", RootMeanSquaredError()])
# model.fit(train_images, train_labels,
#           epochs=3,
#           validation_data=(val_images, val_labels))
# test_metrics = model.evaluate(test_images, test_labels)
# ```

# ### 콜백(callback) 활용

# **콜백**(callback)은 모델 훈련 도중에 부가적으로 호출되는 객체이며
# 학습 과정을 모니터링 하면서 일부 제어기능을 수행하는 다양한 메서드를 제공한다.
# 콜백이 활용되는 주요 기능은 다음과 같다.
# 
# - 모델 체크포인팅: 훈련 중 모델 상태 수시로 저장
# - 훈련 조기 중단: 검증셋 손실이 더 이상 개선되지 않는 경우 훈련 중단
# - 하이퍼 파라미터 조정: 학습률의 동적 변경
# - 훈련 기록 작성: 훈련셋 및 검증셋의 손실값, 평가지표 등 기록 및 시각화
# 
# ```python
# keras.callbacks.ModelCheckpoint
# keras.callbacks.EarlyStopping
# keras.callbacks.LearningRateScheduler
# keras.callbacks.ReduceLROnPlateau
# keras.callbacks.CSVLogger
# ```
# 
# 여기서는 `EarlyStopping`과 `ModelCheckpoint` 두 콜백의 기능을 살펴본다.

# **`fit()` 메서드에서 `callbacks` 인자 사용하기**
# 
# 아래 코드에 사용된 옵션은 다음과 같다.
# 
# - `EarlyStopping`: 검증셋에 대한 정확도가 2 에포크(epoch) 연속 개선되지 않을 때 훈련 종료
# - `ModelCheckpoint`: 매 에포크마다 훈련된 모델 저장. 
#     `save_best_only=True`가 설정된 경우 검증셋에 대한 손실값이 가장 낮은 모델만 저장.

# ```python
# callbacks_list = [
#     keras.callbacks.EarlyStopping(
#         monitor="val_accuracy",
#         patience=2,
#     ),
#     keras.callbacks.ModelCheckpoint(
#         filepath="checkpoint_path.keras",
#         monitor="val_loss",
#         save_best_only=True,
#     )
# ]
# ```

# ```python
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           callbacks=callbacks_list,
#           validation_data=(val_images, val_labels))
# ```

# 조기종료 후 훈련과정에서 저장된 최고 성능의 모델을 불러오면 다음과 같다.

# ```python
# model = keras.models.load_model("checkpoint_path.keras")
# ```

# ### 사용자 정의 콜백 활용

# **`Callback` 클래스 상속**
# 
# 매 에포크와 매 배치 훈련 단계의 시작과 종료 지점에서
# 수행해야 할 기능을 정의해야 하며 아래 메서드를 재정의하는 방식으로 이루어진다.
# 
# ```python
# on_epoch_begin(epoch, logs)
# on_epoch_end(epoch, logs)
# on_batch_begin(batch, logs)
# on_batch_end(batch, logs)
# on_train_begin(logs)
# on_train_end(logs)
# ```
# 
# 각 메서드에 사용되는 인자는 훈련 과정 중에 자동으로 생성된 객체로부터 값을 받아온다.
# 
# - `logs` 인자: 이전 배치와 에포크의 훈련셋과 검증셋에 대한 손실값, 평가지표 등을 포함한 사전 객체.
# - `batch`, `epoch`: 배치와 에포크 정보
# 
# 다음 `LossHistory` 콜백 클래스는 배치 훈련이 끝날 때마다 손실값을 저장하고
# 에포크가 끝날 때마다 배치별 손실값을 그래프로 저장하여 훈련이 종료된 후 시각화하여 보여주도록 한다.

# ```python
# from matplotlib import pyplot as plt
# 
# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs):
#         self.per_batch_losses = []
# 
#     def on_batch_end(self, batch, logs):
#         self.per_batch_losses.append(logs.get("loss"))
# 
#     def on_epoch_end(self, epoch, logs):
#         plt.clf()
#         plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
#                  label="Training loss for each batch")
#         plt.xlabel(f"Batch (epoch {epoch})")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.savefig(f"plot_at_epoch_{epoch}")
#         self.per_batch_losses = []
# ```

# ```python
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           callbacks=[LossHistory()],
#           validation_data=(val_images, val_labels))
# ```

# ### 텐서보드(TensorBoard) 활용

# **텐서보드**(TensorBoard)는 모델 훈련과정을 모니터링하는 최고의 어플이며
# 텐서플로우와 함께 기본적으로 설치된다.
# 
# **주의사항**: 텐서보드 데이터의 저장경로를 
# 
# ```python
# /full_path_to_your_log_dir
# ```
# 
# 대신에 
# 
# ```python
# ./tensorboard_log_dir
# ```
# 
# 등을 사용해야 리눅스, 맥 운영체제에서 오류가 발생하지 않는다.

# ```python
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# 
# tensorboard = keras.callbacks.TensorBoard(
#     log_dir="./tensorboard_log_dir",
# )
# 
# model.fit(train_images, train_labels,
#           epochs=10,
#           validation_data=(val_images, val_labels),
#           callbacks=[tensorboard])
# ```

# 텐서보드를 주피터 노트북에서 아래처럼 실행할 수 있다.

# ```python
# %load_ext tensorboard
# %tensorboard --logdir ./tensorboard_log_dir
# ```

# 텐서보드를 독립적으로 실행하여 훈련과정을 실시간으로 모니터링 하려면
# 아래 명령어를 터미널 창에서 실행하고 반환된 주소로 접속하면 된다.
# 
# ```python
# tensorboard --logdir ./full_path_to_your_log_dir
# ```

# ## 7.4 사용자 정의 훈련 알고리즘: `fit()` 메서드 대체

# ### Training versus inference

# ### Low-level usage of metrics

# ```python
# metric = keras.metrics.SparseCategoricalAccuracy()
# targets = [0, 1, 2]
# predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# metric.update_state(targets, predictions)
# current_result = metric.result()
# print(f"result: {current_result:.2f}")
# ```

# ```python
# values = [0, 1, 2, 3, 4]
# mean_tracker = keras.metrics.Mean()
# for value in values:
#     mean_tracker.update_state(value)
# print(f"Mean of values: {mean_tracker.result():.2f}")
# ```

# ### A complete training and evaluation loop

# **Writing a step-by-step training loop: the training step function**

# ```python
# model = get_mnist_model()
# 
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# optimizer = keras.optimizers.RMSprop()
# metrics = [keras.metrics.SparseCategoricalAccuracy()]
# loss_tracking_metric = keras.metrics.Mean()
# 
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         loss = loss_fn(targets, predictions)
#     gradients = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
# 
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs[metric.name] = metric.result()
# 
#     loss_tracking_metric.update_state(loss)
#     logs["loss"] = loss_tracking_metric.result()
#     return logs
# ```

# **Writing a step-by-step training loop: resetting the metrics**

# ```python
# def reset_metrics():
#     for metric in metrics:
#         metric.reset_state()
#     loss_tracking_metric.reset_state()
# ```

# **Writing a step-by-step training loop: the loop itself**

# ```python
# training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# training_dataset = training_dataset.batch(32)
# epochs = 3
# for epoch in range(epochs):
#     reset_metrics()
#     for inputs_batch, targets_batch in training_dataset:
#         logs = train_step(inputs_batch, targets_batch)
#     print(f"Results at the end of epoch {epoch}")
#     for key, value in logs.items():
#         print(f"...{key}: {value:.4f}")
# ```

# **Writing a step-by-step evaluation loop**

# ```python
# def test_step(inputs, targets):
#     predictions = model(inputs, training=False)
#     loss = loss_fn(targets, predictions)
# 
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs["val_" + metric.name] = metric.result()
# 
#     loss_tracking_metric.update_state(loss)
#     logs["val_loss"] = loss_tracking_metric.result()
#     return logs
# 
# val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
# val_dataset = val_dataset.batch(32)
# reset_metrics()
# for inputs_batch, targets_batch in val_dataset:
#     logs = test_step(inputs_batch, targets_batch)
# print("Evaluation results:")
# for key, value in logs.items():
#     print(f"...{key}: {value:.4f}")
# ```

# ### Make it fast with `tf.function`

# **Adding a `tf.function` decorator to our evaluation step function**

# ```python
# @tf.function
# def test_step(inputs, targets):
#     predictions = model(inputs, training=False)
#     loss = loss_fn(targets, predictions)
# 
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs["val_" + metric.name] = metric.result()
# 
#     loss_tracking_metric.update_state(loss)
#     logs["val_loss"] = loss_tracking_metric.result()
#     return logs
# 
# val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
# val_dataset = val_dataset.batch(32)
# reset_metrics()
# for inputs_batch, targets_batch in val_dataset:
#     logs = test_step(inputs_batch, targets_batch)
# print("Evaluation results:")
# for key, value in logs.items():
#     print(f"...{key}: {value:.4f}")
# ```

# ### Leveraging `fit()` with a custom training loop

# **Implementing a custom training step to use with `fit()`**

# ```python
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# loss_tracker = keras.metrics.Mean(name="loss")
# 
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         inputs, targets = data
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = loss_fn(targets, predictions)
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
# 
#         loss_tracker.update_state(loss)
#         return {"loss": loss_tracker.result()}
# 
#     @property
#     def metrics(self):
#         return [loss_tracker]
# ```

# ```python
# inputs = keras.Input(shape=(28 * 28,))
# features = layers.Dense(512, activation="relu")(inputs)
# features = layers.Dropout(0.5)(features)
# outputs = layers.Dense(10, activation="softmax")(features)
# model = CustomModel(inputs, outputs)
# 
# model.compile(optimizer=keras.optimizers.RMSprop())
# model.fit(train_images, train_labels, epochs=3)
# ```

# ```python
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         inputs, targets = data
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = self.compiled_loss(targets, predictions)
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#         self.compiled_metrics.update_state(targets, predictions)
#         return {m.name: m.result() for m in self.metrics}
# ```

# ```python
# inputs = keras.Input(shape=(28 * 28,))
# features = layers.Dense(512, activation="relu")(inputs)
# features = layers.Dropout(0.5)(features)
# outputs = layers.Dense(10, activation="softmax")(features)
# model = CustomModel(inputs, outputs)
# 
# model.compile(optimizer=keras.optimizers.RMSprop(),
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[keras.metrics.SparseCategoricalAccuracy()])
# model.fit(train_images, train_labels, epochs=3)
# ```
