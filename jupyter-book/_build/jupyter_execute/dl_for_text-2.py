#!/usr/bin/env python
# coding: utf-8

# (ch:nlp)=
# # 자연어 처리

# **감사의 글**
# 
# 아래 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://github.com/fchollet/deep-learning-with-python-notebooks)의 
# 소스코드 내용을 참고해서 작성되었습니다.
# 자료를 공개한 저자에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 여기서 언급되는 코드를
# [(구글 코랩) 자연어 처리](https://colab.research.google.com/github/codingalzi/dlp2/blob/master/notebooks/NB-dl_for_text.ipynb)에서 
# 직접 실행할 수 있다.

# **주요 내용**

# - 단어 주머니
# - 단어 임베딩
# - 트랜스포머

# ## 소개

# 파이썬, 자바, C, C++, C#, 자바스크립트 등 컴퓨터 프로그래밍언어와 구분하기 위해 
# 일상에서 사용되는 한국어, 영어 등을 **자연어**<font size='2'>natural language</font>라 
# 부른다. 
# 
# 자연어의 특성상 정확한 분석을 위한 알고리즘을 구현하는 일은 사실상 매우 어렵다. 
# 딥러닝 기법이 활용되기 이전까지는 적절한 규칙을 구성하여 자연어를 이해하려는 
# 수 많은 시도가 있어왔지만 별로 성공적이지 않았다.

# 1990년대부터 인터넷으로부터 구해진 엄청난 양의 텍스트 데이터에 머신러닝 기법을
# 적용하기 시작했다. 단, 주요 목적이 **언어의 이해**가 아니라 
# 아래 예제들처럼 입력 문장을 분석하여 
# **통계적으로 유용한 정보를 예측**하는 방향으로 수정되었다.
# 
# - 문장 분류: "이 문장의 주제는?"
# - 내용 필터링: "욕설이 포함되었나?"
# - 감성 분석: "내용이 긍정이야 부정이야?"
# - 언어 모델링: "이 문장에 이어 어떤 단어가 있어야 하지?"
# - 번역: "이거를 한국어로 어떻게 말해?"
# - 요약: "이 기사를 한 줄로 요약하면?"
# 
# 이와 같은 분석을 **자연어 처리**<font size='2'>Natural Language Processing</font>라 하며
# 단어, 문장, 문단 등에서 찾을 수 있는 패턴을  인식하려 시도한다. 

# **머신러닝 활용**

# 자연어 처리를 위해 1990년대부터 시작된 머신러닝 활용의 변화과정은 다음과 같다.

# - 1990 - 2010년대 초반: 
#     결정트리, 로지스틱 회귀 모델이 주로 활용됨.
# 
# - 2014-2015: LSTM 등 시퀀스 처리 알고리즘 활용 시작
# 
# - 2015-2017: (양방향) 순환신경망이 기본적으로 활용됨.
# 
# - 2017-2018: 트랜스포머<font size='2'>transformer</font> 모델이 최고의 성능 발휘하며, 
#     많은 난제들을 해결함. 현재 가장 많이 활용되는 모델임.

# ## 텍스트 데이터 전처리

# 딥러닝 모델은 문장 자체를 처리할 수 없다.
# 따라서 문장을 수치형 텐서로 변환하는 **문장 벡터화**<font size='2'>text vectorization</font> 과정이 요구되며
# 보통 다음 세 단계를 따른다.
# 
# 1. **문장 표준화**<font size='2'>text standardization</font>: 소문자화, 마침표 제거 등등
# 1. **토큰화**<font size='2'>tokenization</font>: 기본 단위의 **유닛**<font size='2'>units</font>으로 쪼개기.
#     문자, 단어, 단어 집합 등이 토큰으로 활용됨.
# 1. **어휘 색인화**<font size='2'>vocabulary indexing</font>: 토큰 각각을 하나의 수치형 벡터로 변환.
# 
# 아래 그림은 문장 벡터화의 기본적인 과정을 잘 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-01.png" style="width:60%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 문장 표준화

# 다음 두 문장을 표준화를 통해 동일한 문장으로 변환해보자.
# 
# - "sunset came. i was staring at the Mexico sky. Isnt nature splendid??"
# - "Sunset came; I stared at the M&eacute;xico sky. Isn't nature splendid?"

# 예를 들어 다음과 같은 표준화 기법을 사용할 수 있다.
# 
# - 모두 소문자화
# - `.`, `;`, `?`, `'` 등 특수 기호 제거
# - 특수 알파벳 변환: "&eacute;"를 "e"로, "&aelig;"를 "ae"로 등등
# - 동사/명사의 기본형 활용: "cats"를 "[cat]"로, "was staring"과 "stared"를 "[stare]"로 등등.

# 그러면 위 두 문장 모두 아래 문장으로 변환된다.

# - "sunset came i [stare] at the mexico sky isnt nature splendid"

# 표준화 과정을 통해 어느 정도의 정보를 상실하게 되지만
# 학습해야할 내용을 줄여 일반화 성능이 보다 좋은 모델을 훈련시키는 장점이 있다.
# 하지만 분석 목적에 따라 사용되는 표준화 기법이 달라질 수 있음에 주의해야 한다. 
# 예를 들어 인터뷰 기사의 경우 물음표(`?`)는 제거하면 안된다.

# ### 토큰화

# 문장 표준화 이후 데이터 분석의 기본 단위인 토큰으로 쪼개야 한다.
# 보통 아래 세 가지 방식 중에 하나를 사용한다.
# 
# - 단어 기준 토큰화
#     - 공백으로 구분된 단어들로 쪼개기. 
#     - 경우에 따라 동사 어근과 어미를 구분하기도 함: "star+ing", "call+ed" 등등
# - N-그램 토큰화
#     - N-그램 토큰: 연속으로 위치한 N 개(이하)의 단어 묶음
#     - 예제: "the cat", "he was" 등은 2-그램(바이그램) 토큰이다.
# - 문자 기준 토큰화
#     - 하나의 문자를 하나의 토큰으로 지정.
#     - 문장 생성, 음성 인식 등에서 활용됨.

# 일반적으로 문자 기준 토큰화는 잘 사용되지 않는다. 
# 여기서도 단어 기준과 N-그램 토큰화만 이용한다.
# 
# - 단어 기준 토큰화: 단어들의 순서를 중요시하는 **순차 모델**<font size='2'>sequence models</font>을 사용할 경우 주로 활용된다.
# - N-그램 토큰화: 단어들의 순서를 별로 상관하지 않는 **단어 주머니**<font size='2'>bag-of-words</font> 
#     모델을 사용할 경우 주로 활용된다.
#     - N-그램: 단어들 사이의 순서에 대한 지엽적 정보를 어느 정도 유지한다.
#     - 일종의 특성 공학<font size='2'>feature engineering</font> 기법이며,
#         트랜스포머 등 최신 기법에는 활용되지 않는다.

# 단어 주머니<font size='2'>bag-of-words</font>는 N-그램으로 구성된 집합을 의미하며 
# **N-그램 주머니**라고도 불린다.
# 예를 들어 "the cat sat on the mat." 문장에 대한 
# 2-그램(바이그램) 집합과 3-그램 집합은 각각 다음과 같다.

# - 2-그램(바이그램) 집합
# 
#     ```
#     {"the", "the cat", "cat", "cat sat", "sat",
#     "sat on", "on", "on the", "the mat", "mat"}
#     ```

# - 3-그램 집합
# 
#     ```
#     {"the", "the cat", "cat", "cat sat", "the cat sat",
#     "sat", "sat on", "on", "cat sat on", "on the",
#     "sat on the", "the mat", "mat", "on the mat"}
#     ```

# ### 어휘 색인화

# 일반적으로 먼저 훈련셋에 포함된 모든 토큰들의 색인(인덱스)을 생성한 다음에
# 원-핫, 멀티-핫 인코딩 등의 방식을 사용하여 수치형 텐서로 변환한다.
# 
# {numref}`%s장 <ch:getting_started_with_neural_networks>`에서 언급한 대로 
# 보통 사용 빈도가 높은 2만 또는 3만 개의 단어만을 대상으로 어휘 색인화를 진행한다.
# 당시에 IMDB 영화 후기 데이터셋을 불러올 때
# `num_words=10000`을 사용하여 사용 빈도수가 상위 1만 등 안에 들지 않는 단어는
# 영화 후기에서 무시되도록 하였다.
# 
# ```python
# from tensorflow.keras.datasets import imdb
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# ```

# 케라스의 imdb 데이터셋은 이미 문장 벡터화 전처리가 완료된 상태였다.
# 즉 영화 후기가 정수들의 리스트로 이미 변환된 상태였다.
# 하지만 여기서는 영어 문장으로 작성된 원본 영화후기로 구성된 imdb 데이터셋을 
# 다운로드하여 문장 벡터화 전처리를 직접 수행하는 단계부터 살펴보려 한다.

# **0과 1의 특별한 기능**

# 문장을 변환한 벡터에 사용되는 0과 1은 특별한 기능을 수행한다. 
# 
# - 미등록 어휘 인덱스 기능: 어휘 색인에 미등록된(out-of-vocabulary, OOV) 단어는 모두 1로 지정된다.
#     그런 단어는 일반 문장으로 재번역되는 경우 "[UNK]", 즉 모르는 단어로 처리된다.
# - 패딩 기능: 문장의 길이를 통일시키기 위해 0을 패딩으로 사용한다.
#     0을 **마스크 토큰**<font size='2'>mask token</font>이라 부르기도 하며,
#     모델 훈련과정에서 무시되도록 할 수 있다.
#     
#     ```
#     [[5,  7, 124, 4, 89, 65],
#      [8, 34,  21, 0,  0,  0]]
#     ```

# :::{prf:example} `TextVectorization` 층 활용법
# :label: exp-textvectorization
# 
# 케라스의 `TextVectorization` 층을 이용하여 문장 벡터화를 진행할 수 있다.
# 아래 코드는 `TextVectorization` 층 구성에 사용되는 주요 기본 설정을 보여준다.
# 표준화와 토큰화 방식을 임의로 지정해서 활용할 수도 있지만 여기서는 자세히 다루지 않는다.
# 
# - 표준화: `standardize='lower_and_strip_punctuation'` (소문자화와 마침표 등 제거)
# - 토큰화: `split='whitespace'` (단어 기준 쪼개기), `ngrams=None` (n-그램 미사용)
# - 출력 모드: `output_mode="int"` (정수 인코딩)
# 
# ```python
# >>> from tensorflow.keras.layers import TextVectorization
# 
# >>> text_vectorization = TextVectorization(
# ...     standardize='lower_and_strip_punctuation',  # 기본값
# ...     split='whitespace',                         # 기본값
# ...     ngrams=None,                                # 기본값
# ...     output_mode='int',
# ... )
# ```
# 
# 예를 들어, 아래 데이터셋을 이용하여 문장 벡터화에 사용될 어휘 색인을 생성해보자.
# 
# ```python
# >>> dataset = [
# ...     "I write, erase, rewrite",
# ...     "Erase again, and then",
# ...     "A poppy blooms.",
# ... ]
# ```
# 
# 어휘 색인은 `adapt()` 메서드를 이용하여 만든다.
# 
# ```python
# >>> text_vectorization.adapt(dataset)
# ```
# 
# 생성된 어휘 색인은 다음과 같다.
# 
# ```python
# >>> vocabulary = text_vectorization.get_vocabulary()
# >>> vocabulary
# ['',
#  '[UNK]',
#  'erase',
#  'write',
#  'then',
#  'rewrite',
#  'poppy',
#  'i',
#  'blooms',
#  'and',
#  'again',
#  'a']
# ```
# 
# 생성된 어휘 색인을 활용하여 새로운 문장을 벡터화 해보자.
# 
# ```python
# >>> test_sentence = "I write, rewrite, and still rewrite again"
# >>> encoded_sentence = text_vectorization(test_sentence)
# >>> print(encoded_sentence)
# tf.Tensor([ 7  3  5  9  1  5 10], shape=(7,), dtype=int64)
# ```
# 
# 벡터화된 텐서로부터 문장을 복원하면 표준화된 문장이 생성된다.
# 
# ```python
# >>> inverse_vocab = dict(enumerate(vocabulary))
# >>> decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
# >>> print(decoded_sentence)
# i write rewrite and [UNK] rewrite again
# ```
# :::

# :::{admonition} `TextVectorization` 층과 GPU
# :class: warning
# 
# `TextVectorization` 층은 GPU 또는 TPU에서 지원되지 않는다.
# 따라서 모델 구성에 직접 사용하는 방식은 모델의 훈련을
# 늦출 수 있기에 권장되지 않는다.
# 여기서는 대신에 데이터셋 전처리를 모델 구성과 독립적으로 처리하는 방식을 이용한다.
# 
# 하지만 훈련이 완성된 모델을 실전에 배치할 경우 `TextVectorization` 층을
# 완성된 모델에 추가해서 사용하는 게 좋다.
# :::

# ## 텍스트 데이터 변환

# 훈련시키려 하는 자연어 처리 모델에 따라 문장(텍스트)을 변환하는 방식이 달라진다.
# 
# - 단어 주머니<font size='2'>bag-of-words</font> 사용 모델
#     - 단어들의 순서를 무시. 문장을 단어들의 집합으로 다룸.
#     - 2015년 이전까지 주로 사용됨.
# - 시퀀스<font size='2'>sequence</font> 사용 모델
#     - RNN: 단어들의 순서를 시계열 데이터의 스텝처럼 간주. 2015-2016에 주로 사용됨.
#     - 트랜스포머<font size='2'>Transformer</font> 아키텍처. 
#         기본적으로 순서를 무시하지만 단어 위치를 학습할 수 있는 능력을 가짐.
#         2017년 이후 많이 활용됨.

# 여기서는 IMDB 영화 후기 데이터를 이용하여 두 모델 방식의 
# 활용법과 차이점을 소개한다.

# ### IMDB 영화 후기 데이터셋 준비

# IMDB 데이터셋을 직접 다운로드하여 전처리하는 과정을 자세히 살펴본다. 

# **준비 과정 1: 데이터셋 다운로드 압축 풀기**

# [aclIMDB_v1.tar](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) 파일을 
# 다운로드 한 후에 압축을 풀면 아래 구조의 디렉토리가 생성된다.
# 
# ```
# aclImdb/
# ...train/
# ......pos/
# ......neg/
# ...test/
# ......pos/
# ......neg/
# ```
# 
# `train`의 `pos`와 `neg` 서브디렉토리에 각각 12,500 개의 긍정과 부정 후기가
# 포함되어 있다. `aclImdb/train/unsup` 서브디렉토리는 필요 없기에 삭제한다.

# **준비 과정 2: 검증셋 준비**

# 훈련셋의 20%를 검증셋으로 떼어낸다.
# 이를 위해 `aclImdb/val` 디렉토리를 생성한 후에
# 긍정과 부정 훈련셋 모두 무작위로 섞은 후 그중 20%를 검증셋 디렉토리로 옮긴다.

# **준비 과정 3: 텐서 데이터셋 준비**

# `text_dataset_from_directory()` 함수를 이용하여 
# 훈련셋, 검증셋, 테스트셋을 준비한다. 
# 자료형은 모두 `Dataset`이며, 배치 크기는 32를 사용한다.

# ```python
# batch_size = 32
# 
# train_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/train", batch_size=batch_size
#     )
# 
# val_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/val", batch_size=batch_size
#     )
# 
# test_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/test", batch_size=batch_size
#     )
# ```

# 각 데이터셋은 배치로 구분되며
# 배치의 각 입력 데이터 샘플은 텐서플로우의 문자열 자료형인 `tf.string` 텐서이고, 
# 타깃은 0 또는 1의 `int32` 텐서로 지정된다. 
# 0은 부정을, 1은 긍정을 나타낸다.
# 배치의 크기는 32이다.
# 
# 예를 들어, 첫째 배치의 입력과 타깃 데이터의 정보는 다음과 같다.

# ```python
# >>> for inputs, targets in train_ds:
# ...     # 예제: 첫째 배치의 첫째 후기
# ...     print("inputs[0]:", inputs[0])
# ...     print("targets[0]:", targets[0])    
# ...     break
# inputs[0]: tf.Tensor(b'The film begins with a bunch of kids in reform school and focuses on a kid named \'Gabe\', who has apparently worked hard to earn his parole. Gabe and his sister move to a new neighborhood to make a fresh start and soon Gabe meets up with the Dead End Kids. The Kids in this film are little punks, but they are much less antisocial than they\'d been in other previous films and down deep, they are well-meaning punks. However, in this neighborhood there are also some criminals who are perpetrating insurance fraud through arson and see Gabe as a convenient scapegoat--after all, he\'d been to reform school and no one would believe he was innocent once he was framed. So, when Gabe is about ready to be sent back to "The Big House", it\'s up to the rest of the gang to save him and expose the real crooks.<br /><br />The "Dead End Kids" appeared in several Warner Brothers films in the late 1930s and the films were generally very good (particularly ANGELS WITH DIRTY FACES). However, after the boys\' contracts expired, they went on to Monogram Studios and the films, to put it charitably, were very weak and formulaic--with Huntz Hall and Leo Gorcey being pretty much the whole show and the group being renamed "The Bowery Boys". Because ANGELS WASH THEIR FACES had the excellent writing and production values AND Hall and Gorcey were not constantly mugging for the camera, it\'s a pretty good film--and almost earns a score of 7 (it\'s REAL close). In fact, while this isn\'t a great film aesthetically, it\'s sure a lot of fun to watch, so I will give it a 7! Sure, it was a tad hokey-particularly towards the end when the kids take the law into their own hands and Reagan ignores the Bill of Rights--but it was also quite entertaining. The Dead End Kids are doing their best performances and Ronald Reagan and Ann Sheridan provided excellent support. Sure, this part of the film was illogical and impossible but somehow it was still funny and rather charming--so if you can suspend disbelief, it works well.', shape=(), dtype=string)
# targets[0]: tf.Tensor(1, shape=(), dtype=int32)
# ```

# ### 단어 주머니 활용법

# 단어 주머니에 채울 토큰으로 어떤 N-그램을 사용할지 먼저 지정해야 한다. 
# N-그램의 종류는 다음과 같다.
# 
# - 유니그램(unigrams): 하나의 단어가 하나의 토큰
# - N-그램(N-grams): 최대 N 개의 이어지는 단어로 이루어진 문구가 하나의 토큰

# **방식 1: 유니그램 멀티-핫 인코딩**

# 예를 들어, "the cat sat on the mat" 문장을 유니그램으로 처리하면 다음 
# 단어 주머니가 생성된다.
# 
# ```
# {"cat", "mat", "on", "sat", "the"}
# ```
# 
# 그런 다음 어휘 색인을 이용하여 생성된 단어 주머니를 멀티-핫 인코딩을 이용하여 1차원 이진 텐서, 
# 즉 0과 1로만 구성된 벡터로 변환한다.

# 유니그램 멀티-핫 인코딩을 `TextVectorization` 클래스를 이용하면 간단하게 처리할 수 있다.
# 
# - `max_tokens=20000` 옵션: 빈도가 20,000 등 안에 드는 단어만 멀티-핫 인코딩 적용
# - `output_mode="multi_hot"` 옵션: 유니그램을 멀티-핫 인코딩하기

# ```python
# text_vectorization = TextVectorization(
#     max_tokens=20000,
#     output_mode="multi_hot",
# )
# ```

# 이제 훈련셋의 후기에 포함된 단어들을 이용하여 어휘 색인을 생성한다.
# 이를 위해 훈련셋에 포함된 후기만 따로 떼어낸다.
# 이 데이터셋은 이후에도 다른 종류의 어휘 색인을 생성하는 데에 계속 사용된다.

# ```python
# # 어휘 색인 생성 대상 훈련셋 후기 문장 데이터셋
# text_only_train_ds = train_ds.map(lambda x, y: x)
# ```

# 이제 `adapt()` 메서드를 이용하여 유니그램 토큰을 멀티-핫 인코딩 하기 위해 필요한 어휘 색인을 생성한다.

# ```python
# # 어휘색인 생성
# text_vectorization.adapt(text_only_train_ds)
# ```

# 생성된 어휘 색인을 이용하여 훈련셋, 검증셋, 테스트셋 모두 멀티-핫 인코딩, 즉 벡터화한다. 

# ```python
# binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
# binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
# binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
# ```

# 첫째 배치의 첫째 샘플의 입력과 타깃 데이터는 다음과 같다.
# `max_tokens=20000`으로 지정하였기에 모든 문장은 길이가 2만인 멀티-핫 벡터로 변환된다.
# 예를 들어, n 번 인덱스의 값이 1이면 인덱스에 해당하는 단어가 영화 후기에 사용되었음을 나타낸다.

# ```python
# 첫째 배치의 첫째 데이터 입력값: tf.Tensor([1. 1. 1. ... 0. 0. 0.], shape=(20000,), dtype=float32)
# 첫째 배치의 첫째 데이터 타깃:   tf.Tensor(0, shape=(), dtype=int32)
# ```

# *밀집 모델 활용*

# 단어 주머니 형식으로 전처리된 데이터를 이용하는 모델은 기본적으로 
# 밀집 모델을 사용한다. 
# 이유는 단어 주머니에 포함된 토큰들 사이에는 순서가 없기에
# 모델도 굳이 순서를 고려할 필요가 없기 때문이다.
# 
# 아래 `get_model()` 함수는 단어 주머니로 전처리된 데이터셋을 이용하는 경우
# 활용될 매우 단순하며 컴파일된 밀집 모델을 반환한다.
# 모델의 출력값은 긍정일 확률이며, 
# 최상위 층의 활성화 함수로 `sigmoid`를 사용한다.

# ```python
# def get_model(max_tokens=20000, hidden_dim=16):
#     inputs = keras.Input(shape=(max_tokens,))
#     x = layers.Dense(hidden_dim, activation="relu")(inputs)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)  # 긍정일 확률 계산
#     
#     model = keras.Model(inputs, outputs)
#     
#     model.compile(optimizer="rmsprop",
#                   loss="binary_crossentropy",
#                   metrics=["accuracy"])
#     
#     return model
# ```

# 밀집 모델 훈련과정은 특별한 게 없다.
# 훈련 후 테스트셋에 대한 정확도가 89% 보다 조금 낮게 나온다.
# 최고 성능의 모델이 테스트셋에 대해 95% 정도 정확도를 내는 것보다는 낮지만
# 무작위로 찍는 모델보다는 훨씬 좋은 모델이다.

# **방식 2: 바이그램 멀티-핫 인코딩**

# 바이그램(2-grams)을 유니그램 대신 이용해보자. 
# 예를 들어 "the cat sat on the mat" 문장을 바이그램으로 처리하면 다음 
# 단어 주머니가 생성된다.
# 
# ```
# {"the", "the cat", "cat", "cat sat", "sat",
#  "sat on", "on", "on the", "the mat", "mat"}
# ```

# 바이그램 멀티-핫 인코딩을 `TextVectorization` 클래스를 이용하면 간단하게 처리할 수 있다.
# 
# - `ngrams=2` 옵션 활용

# ```python
# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="multi_hot",
# )
# ```

# 어휘 색인 생성과 훈련셋, 검증셋, 테스트셋의 벡터화 과정은 동일하다.

# ```python
# text_vectorization.adapt(text_only_train_ds)
# 
# binary_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
# binary_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
# binary_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
# ```

# 이전과 동일한 밀집 모델을 훈련시킨 후 테스트셋에 대한 정확도가 90%를 조금 웃돌 정도로 많이 향상되었다.

# **방식 3: 바이그램 TF-IDF 인코딩**

# N-그램을 벡터화할 때 사용 빈도를 함께 저장하는 방식을 사용할 수 있다.
# 단어의 사용 빈도가 아무래도 문장 평가에 중요한 역할을 수행할 것이기 때문이다.
# 아래 코드에서처럼 `output_mode="count"` 옵션을 사용하면 된다.

# ```python
# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="count"
# )
# ```

# 그런데 이렇게 하면 "the", "a", "is", "are" 등의 사용 빈도는 매우 높은 반면에
# "Chollet" 등의 단어는 빈도가 거의 0에 가깝게 나온다.
# 또한 생성된 벡터의 대부분은 0으로 채워질 것이다. 
# `max_tokens=20000`을 사용한 반면에 하나의 문장엔 많아야 몇 십 개 정도의 단어만 사용되었기 때문이다. 
# 
# ```python
# inputs[0]: tf.Tensor([1. 1. 1. ... 0. 0. 0.], shape=(20000,), dtype=float32)
# ```

# 이 점을 고려해서 사용 빈도를 정규화한다. 
# 평균을 원점으로 만들지는 않고 TF-IDF 값으로 나누기만 실행한다.
# 이유는 평균을 옮기면 벡터의 대부분의 값이 0이 아니게 되어
# 훈련에 보다 많은 계산이 요구되기 때문이다. 
# 
# **TF-IDF**의 의미는 다음과 같다.
# 
# - `TF`(Term Frequency)
#     - 하나의 문장에서 사용되는 단어의 빈도
#     - 높을 수록 중요
#     - 예를 들어, 하나의 후기에 "terrible" 이 많이 사용되었다면
#         해당 후기는 부정일 가능성 높음.
# - `IDF`(Inverse Document Frequency)
#     - 데이터셋 전체 문장에서 사용된 단어의 빈도
#     - 낮을 수록 중요. 
#     - "the", "a", "is" 등의 `IDF` 값은 높지만 별로 중요하지 않음.
# - `TF-IDF = TF/IDF`

# `output_mode="tf_idf"` 옵션을 사용하면 TF-IDF 인코딩을 지원한다.

# ```python
# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="tf_idf",
# )
# ```

# 어휘색인 생성과 훈련셋, 검증셋, 테스트셋의 벡터화 과정은 동일하다. 

# ```python
# text_vectorization.adapt(text_only_train_ds)
# 
# tfidf_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
# tfidf_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
# tfidf_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
# ```

# 훈련 후 테스트셋에 대한 정확도가 다시 89% 아래로 내려간다.
# 여기서는 별 도움이 되지 않았지만 많은 문장 분류 모델에서는 1% 정도의 성능 향상을 가져온다.

# :::{admonition} 문자열을 입력받는 모델 실전 배치하기
# :class: info
# 
# 훈련된 모델을 실전에 배치하려면 문장 벡터화 기능을 전처리 과정에 포함시켜야 한다.
# 이를 위해 `TextVectorization` 층의 결과를 재활용만 하면 된다.
# 
# ```python
# inputs = keras.Input(shape=(1,), dtype="string")
# # 문장 벡터화 추가
# processed_inputs = text_vectorization(inputs)
# # 훈련된 모델에 적용
# outputs = model(processed_inputs)
# 
# # 최종 모델
# inference_model = keras.Model(inputs, outputs)
# ```
# 
# 이제 위 `inference_model`은 일반 문장을 직접 인자로 받을 수 있다.
# 예를 들어 "That was an excellent movie, I loved it."라는 후기는
# 긍정일 확률이 매우 높다고 예측된다.
# 
# ```python
# raw_text_data = tf.convert_to_tensor([
#     ["That was an excellent movie, I loved it."],
# ])
# 
# predictions = inference_model(raw_text_data)
# ```
# :::

# ### 시퀀스 활용법

# 바이그램, 3-그램 등을 이용하여 단어들 사이의 순서 정보를 함께 활용하면 기본적으로 모델의 훈련 성능이 향상된다.
# 하지만 이와 같은 방식은 일종의 특성 공학<font size='2'>feature engineering</font>이며,
# 딥러닝은 어떤 형식의 특성 공학도 하지 않는 방향으로 발전해왔다.
# 
# 여기서는 양방향 LSTM 층으로 구성된 순환 신경망 모델을 훈련시키기 위해
# 문장에 포함된 단어들의 순서를 있는 그대로 전달하는 방식을 알아 본다.
# 훈련셋은 IMDB 영화 후기 데이터셋이다.

# **정수 벡터 데이터셋 준비**

# 먼저 훈련셋의 모든 후기 문장을 정수들의 벡터로 변환한다.
# 단, 후기 문장이 최대 600 개의 단어만 포함하도록 한다. 
# 또한 사용되는 어휘는 빈도 기준 최대 2만 개로 제한한다. 
# 후기의 길이를 600 개의 단어로 제한한 이유는 후기가 평균적으로 233 개의 단어를 사용하고,
# 600 단어 이상을 사용하는 후기는 전체의 5% 정도에 불과하기 때문이다.

# 모든 후기는 최대 600 개의 단어만 포함하도록 하며, 그보다 긴 문장은 잘라서 사용한다.
# 따라서 생성되는 정수들의 벡터는 모두 길이가 600으로 지정된다.
# 후기가 600 개보다 적은 수의 단어를 사용한다면 마스크 토큰 0으로 채운다.
# 벡터에 사용된 정수는 2만보다 작은 값인데,
# 이는 빈도가 가장 높은 2만 개의 단어만을 대상으로 했기 때문이다.
# 즉, 사용 빈도가 너무 낮은 단어는 무시한다.

# ```python
# max_length = 600
# max_tokens = 20000
# 
# text_vectorization = layers.TextVectorization(
#     max_tokens=max_tokens,              # 후기에 사용되는 단어의 종류는 총 2만 종류
#     output_mode="int",
#     output_sequence_length=max_length,  # 하나의 후기에 포함된 최대 단어는 최대 600 개
# )
# 
# # 어휘 색인
# text_vectorization.adapt(text_only_train_ds)
# 
# # 후기를 길이가 2만인 정수들의 리스트로 변환
# int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
# int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
# int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
# ```

# 예를 들어, 변환된 첫째 배치의 첫째 입력 데이터 샘플은 다음과 같다.

# ```python
# tf.Tensor(
# [   11     7     4  8614    18    38     9   139   138   197   640    12
#     30    22   167     6  3035     2    86  3146   664    19    12   291
#     11    14  2400  2996    13    55   322   429    11    19   172     4
#    337    35   116   230   172     4  1107     2   196  1562    14    12
#     10   399     9   100     9    14   478    46  1368   162    31    47
#    509    56     2  7585   645    66   733     5   239  1428     1    17
#      2    86    18     3    56    47   645    12    23    66     6    28
#    920     6   376    19   197   107 14487    39     8  8227    83    23
#    103   235     1    16  6307    13     4   309   869    21     2  7585
#    645    10    14   776     6   158    12   593     5     2   645    67
#     41  3488  5321     8   188    48    67   208    57     1    31    32
#      2  1990    67   154   239  1265    35   154    66     4     1     3
#     67   208     8    50  1244   450    39    55   322     6   103    12
#    217    53     6   493    72   167     6     2  3925     3    11    18
#      7   479     8   144     1    13  8499    49   330     2   223    14
#   5673    22   730    15  1428    15     8     2    86    42   327    18
#     19   943     5   250    16     2   322    57  2027  1932   383    62
#     14     4 13077    16    70     4   110   215    19   157   100   609
#      2  1013     5     1   500    55   322  3987    22   242     4  3852
#    690    14  2207    16    12  2227    13    32     8    32   450   129
#     11     7     4    84    18    16   322     5    98   588    29   172
#   1319  2224     6   381    99   104    10   328    22     6    28  2012
#   2677    19   193    66     6  1810    58     3   460   127     2   247
#    301     4   163    93    12    67   324     1    72   848    19   321
#   2224     6   544     2   698   301    11    29   450   129  1245   183
#    574   149    23   225   158    12    23   341     9   100     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0], shape=(600,), dtype=int64)
# ```

# #### 원-핫 단어 벡터

# 정수 벡터를 순차 모델의 입력값으로 그대로 사용할 수는 없다.
# 대신 정수 벡터에 포함된 각각의 정수에 원-핫 인코딩을 적용하면
# 원-핫 벡터로 이루어진 시퀀스가 생성되고 이를 순환 신경망 모델의 입력값으로 사용한다.

# :::{prf:example} `tf.one_hot()` 함수
# :label: exp-one-hot-sequence
# 
# 정수 벡터 `[2, 1, 4, 0, 0]`의 각 항목에 원-핫 인코딩을 적용하여 
# 생성되는 원-핫 벡터로 이루어진 시퀀스가 생성된다.
# 단, 벡터에 사용된 정수는 0에서 4까지라고 가정한다.
# 
# 따라서 원-핫 벡터 하나의 길이는 5가 되며, 
# `[2, 1, 4, 0, 0]`의 각 항목을 원-핫 벡터로 변환한 값들로 이뤄진 시퀀스는 다음과 같으며,
# `tf.one_hot()` 함수를 이용하면 된다.
# 
# ```python
# >>> tf.one_hot(indices=[2, 1, 4, 0, 0], depth=5)
# [[0, 0, 1, 0, 0],      # 2
#  [0, 1, 0, 0, 0],      # 1
#  [0, 0, 0, 0, 1],      # 4
#  [1, 0, 0, 0, 0],      # 0
#  [1, 0, 0, 0, 0]]      # 0
# ```
# :::

# 아래 코드는 `tf.one_hot()` 함수를 전처리로 활용하는 순환 신경망 모델을 정의한다.
# 이렇게 하면 정수 벡터를 바로 모델의 입력값으로 사용할 수 있다.

# ```python
# inputs = keras.Input(shape=(None,), dtype="int64")
# # 원-핫 인코딩
# embedded = tf.one_hot(inputs, depth=max_tokens)  # (600, 20000) 모양의 출력값 생성
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# 
# model = keras.Model(inputs, outputs)
# ```

# 그런데 위 모델의 훈련 과정이 매우 느리다. 
# 이유는 입력 데이터가 너무 많은 특성을 갖기 때문이다. 
# 입력 데이터 하나, 하나의 문장을 변환시킨 원-핫 단어 벡터의 모양은 `(600, 20000)`인 2차원 텐서이다.
# 따라서 총 `600 * 20,000 = 12,000,000` 개의 항목이 포함되어 있고 그만큼의 
# 가중치가 훈련되어야 한다.
# 이에 더해 양방향 LSTM은 엄청난 양의 반복을 실행하기에 그만큼 훈련 시간이 길어질 수밖에 없다.
# 
# 게다가 훈련된 모델의 성능이 별로 좋지 않다. 
# 테스트셋에 대한 정확도가 87% 정도에 불과해서 바이그램 모델보다 성능이 낮다.

# #### 단어 임베딩

# 앞서 보았듯이 정수 벡터를 원-핫 인코딩해서 원-핫 워드 벡터로 구성된 시퀀스를 생성하는 방식은 별로 적절하지 않다. 
# 변환된 벡터의 특성이 너무 많으며, 단어들 사이의 다음과 같은 유기적 관계도 제대로 반영되지 못한다.
# 
# - "movie"와 "film", "비디오"와 "동영상", "강아지"와 "개" 등이 사실상 동일하다는 동의어 관계
# - "왕"(남자)과 "여왕"(여자), "boy"와 "girl" 등의 성별 관계
# - "king"의 복수는 "kings" 등 문법 규칙 관계
# - "고양이"와 "호랑이"는 고양이과, "개"와 "늑대"는 개과, "고양이"와 "개"는 애완동물, "늑대"와 "호랑이"는 야생동물 등의 생물 분류 관계

# 반면에 **단어 임베딩**<font size='2'>word embedding</font>은 
# 단어들 사이의 유기적 관계를 모델 스스로 학습을 통해 찾도록 유도하며,
# 보통 다음 두 가지 방식으로 활용된다.
# 
# - 모델 훈련과 단어 임베딩 학습 동시 진행: 
#     언어 종류와 모델 훈련 목적에 따라 기본적으로 서로 다른 단어 사이의 관계가 학습되어야 한다.
#     예를 들어, 영화 후기 감성 분석과 재판 판결문 분석은 서로 다른 단어 임베딩을 사용할 수밖에 없다.
# - 기존에 잘 훈련된 워드 임베딩 활용:
#     합성곱 신경망에서 이미지넷 등의 대용량 데이터셋을 활용하여 잘 훈련된 모델을 재활용하는 것처럼 
#     잘 구성된 대용량의 어휘 색인을 이용하여 생성된 단어 임베딩을 활용한다.

# 먼저 케라스의 `Embedding` 층을 모델에 추가하여 단어 임베딩 학습을
# 모델 훈련과 함께 진행하는 방식을 살펴본다.

# **케라스의 `Embedding` 층 활용**

# 케라스의 `Embedding` 층은 일종의 사전처럼 작동하며, 
# 하나의 문장이 입력되면 문장에 포함된 단어들 사이에 존재하는 숨은 연관성을 학습해서 담는
# 벡터로 이루어진 시퀀스를 생성한다.
# 벡터는 부동소수점으로 구성된다.
# 단어들 사이의 연관성은 256 개, 512 개, 1024 개 등 지정된 개수만큼 찾도록 한다.
# 즉, 각각의 단어를 길이가 256, 512, 또는 1024 크기의 벡터로 변환한다.
# 
# 아래 그림은 원-핫 인코딩 방식과 단어 임베딩 방식의 차이점을 보여준다. 
# 
# | 원-핫 단어 벡터 | 단어 임베딩 |
# | :--- | :--- |
# | 희소 벡터 | 밀집 벡터 |
# | 매우 많은 특성 | 적절한 양의 특성 |
# | 직접 전처리 | 모델 스스로 훈련 |

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-02.png" style="width:60%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 예를 들어, 600 개의 단어로 이루어진 문장이 주어졌을 때, 
# 문장에 포함된 각 단어에 대해 다른 단어들과의 연관성을 256 개 찾아낸다.
# 그런 다음 길이가 256인 600개의 벡터로 이루어진 시퀀스, 
# 즉 `(600, 256)` 모양의 시퀀스를 생성한다.

# 그런데 후기 문장의 길이가 600이 되지 않는 경우 나머지는 
# **패딩**<font size='2'>padding</font>에 의해 0으로 채워진다.
# 하지만 이렇게 의미 없이 추가된 0이 훈련에 좋지 않은 영향을 미친다.
# 따라서 모델이 패딩으로 사용된 0이 있다는 사실을 인지하도록 도와주는 
# **마스킹**<font size='2'>masking</font> 기능을 활용하면 보다 좋은 성능을 얻을 수 있다.

# 방금 설명한 것을 아래 `Embedding` 층이 실행한다. 
# 
# ```python
# layers.Embedding(input_dim=20000, output_dim=256, mask_zero=True)
# ```

# :::{admonition} `Embedding` 층과 `mask_zero` 옵션
# :class: warning
# 
# 입력 벡터에 사용되는 0이 특별한 의미를 갖는 경우 `mask_zero` 옵션을 기본값인 `False`로 지정해야 한다.
# `mask_zero=True`는 입력 벡터의 길이가 달라 0을 패딩으로 사용해야 하는 경우에만 사용한다.
# :::

# 아래 코드는 양방향 LSTM을 사용하는 순환 신경망 모델에 
# 마스킹을 활용 단어 임베딩 방식을 적용하는 것을 보여준다.

# ```python
# inputs = keras.Input(shape=(None,), dtype="int64")
# # 마스킹 활용 단어 임베딩
# embedded = layers.Embedding(
#     input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
# # 양방향 LSTM
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# 
# model = keras.Model(inputs, outputs)
# ```

# `Embedding` 층을 통과하면 후기 문장에 사용된 600개 단어 각각이 총 256개의 특성으로 구성된
# 벡터로 변환되었음을 `summary()` 를 통해 확인할 수 있다.

# ```python
# >>> model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_7 (InputLayer)        [(None, 600)]            0         
#                                                                  
#  embedding_1 (Embedding)     (None, 600, 256)         5120000   
#                                                                  
#  bidirectional_2 (Bidirectio  (None, 64)               73984     
#  nal)                                                            
#                                                                  
#  dropout_5 (Dropout)         (None, 64)                0         
#                                                                  
#  dense_8 (Dense)             (None, 1)                 65        
#                                                                  
# =================================================================
# Total params: 5,194,049
# Trainable params: 5,194,049
# Non-trainable params: 0
# ```

# 모델의 훈련은 원-핫 인코딩 방식보다 훨씬 빠르게 이루어지며 성능은 88% 정도로 살짝 향상된다. 
# 바이그램 모델보다 성능이 여전히 떨어지는 이유 중에 하나는 후기에 사용된 단어의 수를 600 개로 제한하였기 때문이다.

# #### GloVe 단어 임베딩

# 합성곱 신경망에서 이미지넷<font size='2'>ImageNet</font> 데이터셋을 이용하여 
# 잘 훈련된 모델을 재활용했던 것처럼 대량의 문장을 분석하여 훈련된 단어 임베딩을 활용할 수 있다.
# 
# 여기서는 2014년에 스탠포드 대학교의 연구자들이 위키피디아에 사용된 수 많은 문장을 분석하여 생성한 
# [GloVe(Gloval Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/)
# 단어 임베딩을 활용한다.

# **임베딩 파일 다운로드**

# [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip) 파일을 다운로드한다. 
# 압축파일의 크기는 822 MB이다.
# 다운로드한 파이릐 압축을 풀면 다음 세 개의 파일이 생성된다.
# 
# - `glove.6B.50d.txt` (약 1 GB)
# - `glove.6B.100d.txt` (약 670 MB)
# - `glove.6B.200d.txt` (약 330 MB)
# - `glove.6B.300d.txt` (약 160 MB)

# 예를 들어 `glove.6B.100d.txt` 파일의 첫 20줄은 다음과 같다.
# 단어별로 100개의 부동소수점이 지정되어 있지만 화면상 10여 개의 값만 보여지고 나머지는 화면에서 잘렸다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/dlp2/master/jupyter-book/imgs/ch11-glove01.jpg" style="width:80%;"></div>

# **임베딩 층 지정**

# 총 40만 개의 단어 정보를 포함한 `glove.6B.100d.txt` 파일을 이용하여 단어 임베딩을 실행하려면
# 임베딩 층을 다음과 같이 설정한다.
# 
# - `max_tokens = 20000`: 빈도가 2만등 안에 드는 단어만 대상으로 함.
# - `embedding_dim = 100`: 단어별로 총 100개의 특성을 갖는 벡터가 연결된다.
# - `embedding_matrix`: `glove.6B.100d.txt` 파일을 참고하여 생성된 `(20000, 100)` 모양의 임베딩 행렬이다.
#     이 행렬을 참조하여 후기에 사용된 각 단어에 대해 하나의 벡터를 연결한다.
#     즉, 최대 600 개의 단어로 구성된 문장이 입력되면 해당 문장을 `(600,100)` 모양의 시퀀스로 변환한다.
# - `Constant`: 단어 임베딩의 기준으로 사용될 행렬을 상수로 지정한다.
# - `trainable=False`: 단어 임베딩의 기준이 모델 훈련 과정 내내 고정시킨다. 
#     즉, 단어 임베딩에 대한 별도 훈련은 시키지 않는다.

# ```python
# embedding_layer = layers.Embedding(
#     max_tokens,
#     embedding_dim,
#     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#     trainable=False,
#     mask_zero=True,
# )
# ```

# **모델 지정 및 훈련**

# 모델 구성은 임베딩 층만 제외하고 이전과 동일하다.

# ```python
# inputs = keras.Input(shape=(None,), dtype="int64")
# # GloVe 단어 임베딩 활용
# embedded = embedding_layer(inputs)
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# ```

# `Embedding` 층을 통과하면 후기 문장에 사용된 600개 단어 각각이 총 100개의 특성으로 구성된
# 벡터로 변환되었음을 `summary()` 를 통해 확인할 수 있다.

# ```python
# >>> model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_8 (InputLayer)        [(None, 600)]            0         
#                                                                  
#  embedding_2 (Embedding)     (None, 600, 100)         2000000   
#                                                                  
#  bidirectional_3 (Bidirectio  (None, 64)               34048     
#  nal)                                                            
#                                                                  
#  dropout_6 (Dropout)         (None, 64)                0         
#                                                                  
#  dense_9 (Dense)             (None, 1)                 65        
#                                                                  
# =================================================================
# Total params: 2,034,113
# Trainable params: 34,113
# Non-trainable params: 2,000,000
# _________________________________________________________________
# ```

# 훈련 결과가 특별히 나아지지는 않는다. 이유는 훈련 데이터셋이 충분히 크기에 굳이 외부 임베딩 파일을
# 이용하지 않아도 자체적으로 단어들 사이의 관계를 잘 찾기 때문이다.
# 하지만 훈련셋이 더 작은 경우 잘 훈련된 임베딩 파일을 이용하면 보다 효율적으로 훈련이 진행된다.

# ## 트랜스포머 아키텍처

# 2017년 논문 ["Attention is all you need"](https://arxiv.org/abs/1706.03762)에서
# 소개된 **트랜스포머**<font size='2'>Transformer</font> 아키텍처는 자연어처리 분야에서 혁명을 불러왔다.
# 트랜스포머는 "**뉴럴 어텐션**<font size='2'>Neural Attention</font>" 기법을 이용하여 
# 순환층과는 다르게 작동하는 순차 모델<font size='2'>sequence model</font>을 지원한다.
# 
# 여기서는 뉴럴 어텐션의 작동법을 설명한 후에 트랜스포머 인코더를 이용하여 IMDB 영화 후기 모델을 구현한다. 

# ### 셀프 어텐션

# 입력값의 특성 중에 보다 중요한 특성에 **집중**<font size='2'>attention</font>하면 보다 효율적으로
# 훈련이 진행될 수 있다. 
# 아래 그림이 집중이 어떻게 활용되는지 잘 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-05.png" style="width:70%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 앞서 유사한 아이디어를 활용한 적이 있다.
# 
# - 합성곱 신경망의 맥스 풀링<font size='2'>max pooling</font>: 지역적으로 가장 중요한 특성만 사용한다.
# - TF-IDF 정규화: 텍스트 벡터화를 위해 사용되는 TF-IDF 정규화는
#     사용되는 토큰에 포함된 정보의 중요도를 평가하여 보다 중요한 정보를 담은 토큰에 집중한다. 

# 자연어 처리에서의 **셀프 어텐션**<font size='2'>self-attention</font> 기법은 
# 주어진 문장에 사용된 단어들의 연관성을 평가하여, 즉 **문맥**<font size='2'>context</font>을 파악하여
# 문장을 시퀀스로 변환하는 데에 사용된다.
# 
# 아래 그림은 "The train  left the station on time." 이라는 문장에
# 셀프 어텐션을 적용하여 입력값을 변환하는 과정을 보여준다.
# 
# - 1단계: 문장에 사용된 각 토큰들 사이의 연관성 계산.
# - 2단계: 계산된 연관성을 (토큰) 벡터와 결합시킨 후 새로운 토큰 벡터들의 시퀀스 생성.
#     아래 그림에서는 "station" 단어에 해당하는 벡터가 변환되는 과정을 보여줌.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-06.png" style="width:100%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **질문-열쇠-값 모델**

# 셀프 어텐션의 작동 과정을 식으로 표현하면 다음과 같다. 
# 
#     outputs = sum(inputs * pairwise_scores(inputs, inputs))
#                     |                        |      |
#                    (C)                      (A)    (B)
# 
# 위 식은 원래 검색 엔진 또는 추천 시스템에 사용되는 
# 보다 일반화된 셀프 어텐션의 작동과정을 표현한 식의 특별한 경우를 보여준다. 
# 
#     outputs = sum(values * pairwise_scores(query, keys))
#                     |                        |      |
#                    (C)                      (A)    (B)

# 예를 들어, 아래 그림은 "dogs on the beach." **질문**<font size='2'>query</font>에 가장 
# 적절한 사신을 검색한다면 각 사진과의 **핵심 연관성**<font size='2'>key</font> 점수를
# 해당 사진<font size='2'>value</font>과 결합하여 가장 높은 점수를 갖는 사진을
# 추천하는 것을 보여준다. 

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-07.png" style="width:80%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# 질문-열쇠-값<font size='2'>query-key-value</font> 모델을 실전에 적용할 때 
# 열쇠<font size='2'>key</font>와 값<font size='2'>value</font>이 동일한 경우가 많다. 
# 
# - 기계 번역:
#     "How's the weather today?"를 스페인어로 기계 번역하려 할 경우
#     스페인어로 날씨에 해당하는 "tiempo"를 열쇠로 해서 주어진 영어 문장(query)에 
#     사용된 단어들과 비교해야 한다. 
# - 텍스트 분류:
#     앞서 셀프 어텐션 설명을 위해 사용된 그림에서처럼 query, keys, values 모두 동일하다.

# ### 멀티헤드 어텐션

# 단어들 사이의 연관성을 다양한 방식으로 알아내기 위해 셀프 어텐션을 수행하는 
# **헤드**<font size='2'>head</font>를 여러 개 병렬로 처리한 후에 다시 합치는 기법을
# **멀티헤드 어텐션**<font size='2'>multi-head attention</font>이다.
# 아래 그림은 두 개의 헤드를 사용하는 것을 보여주며 각각의 헤드가 하는 일은 다음과 같다.
# 
# - 질문, 키, 값을 각각 서로 다른 밀집 밀집 층으로 구성된 블록을 통과 시킨다.
# - 이후 변환된 질문, 키, 값에 셀프 어텐션을 적용한다.
# 
# 각 헤드에 포함된 밀집 층 블록으로 인해 멀티헤드 어텐션 층에서도
# 학습이 이루어진다.
# 
# **참고**: 채널 분리 합성곱 층의 알고리즘과 기본 아이디어가 유사하다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-08.png" style="width:70%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 트랜스포머 인코더

# 멀티헤드 어텐션을 밀집<font size='2'>dense</font> 층, 정규화 층, 잔차 연결 등과 조합하여
# **트랜스포머 인코더**<font size='2'>transformer encoder</font>를 생성한다.
# 
# 아래 그림에서 사용되는 정규화 층인 케라스의 `LayerNormalization`은
# 정규화를 배치 단위가 아닌 시퀀스 단위로 정규화를 실행한다.
# 시퀀스 데이터를 처리할 때는 `BatchNormalization` 보다 잘 작동한다. 

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-09.png" style="width:35%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# **트랜스포머 구현**

# 위 그림에서 설명된 트랜스포머 인코더를 층으로 구현하면 다음과 같다.
# 생성자의 입력값을 예를 들어 설명하면 다음과 같다.
# 
# - `embed_dim`
#     - 트랜스포머 인코더는 "단어 임베딩" 층을 통과한 값을 받음.
#     - 예를 들어 `embed_dim=256`은 단어 임베딩이
#         `(600, 256)` 모양의 샘플을 생성할 것을 기대함.
# - `dense_dim`: 밀집 층에서 사용되는 유닛(unit) 수
# - `num_heads`: 헤드(head) 수

# ```python
# class TransformerEncoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim)
#         self.dense_proj = keras.Sequential(
#             [layers.Dense(dense_dim, activation="relu"),
#              layers.Dense(embed_dim),]
#         )
#         self.layernorm_1 = layers.LayerNormalization()
#         self.layernorm_2 = layers.LayerNormalization()
# 
#     def call(self, inputs, mask=None):
#         if mask is not None:
#             mask = mask[:, tf.newaxis, :]
#         attention_output = self.attention(
#             inputs, inputs, attention_mask=mask)
#         proj_input = self.layernorm_1(inputs + attention_output)
#         proj_output = self.dense_proj(proj_input)
#         return self.layernorm_2(proj_input + proj_output)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "dense_dim": self.dense_dim,
#         })
#         return config
# ```

# **트랜스포머 인코더 활용 모델**

# 훈련 데이터셋이 입력되면 먼저 단어 임베딩을 이용하여 
# 단어들 사이의 연관성을 찾는다.
# 이후 트랜스포머 인코더로 셀프 어텐션을 적용한다.
# 
# 사용되는 변수들은 다음과 같다.
# 
# - `vocab_size = 20000`: 어휘 색인 크기
# - `embed_dim = 256`: 단어 임베딩 특성 수
# - `dense_dim = 32`: 트랜스포머 인코더에 사용되는 밀집층의 유닛(unit) 수
# - `num_heads = 2`: 트랜스포머 인코더에 사용되는 밀집층의 헤드(head) 수

# ```python
# vocab_size = 20000
# embed_dim = 256
# num_heads = 2
# dense_dim = 32
# 
# inputs = keras.Input(shape=(None,), dtype="int64")
# 
# x = layers.Embedding(vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# 
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dropout(0.5)(x)
# 
# outputs = layers.Dense(1, activation="sigmoid")(x)
# 
# model = keras.Model(inputs, outputs)
# 
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# 
# model.summary()
# ```

# 훈련 과정은 특별한 게 없다.
# 테스트셋에 대한 정확도가 87.5% 정도로 바이그램 모델보다 좀 더 낮다.

# **모델 비교**

# 자연어처리와 관련된 모델을 단어 순서 인식과 문맥 이해 차원에서 비교하면 다음과 같다.
# 
# | | 단어순서 인식 | 문맥 이해 |
# | :---: | :---: | :---: |
# | 유니그램 주머니 모델 | X | X |
# | 바이그램 주머니 모델 | $\triangle$ | X |
# | RNN | O | X |
# | 셀프 어텐션 | X | O |
# | 트랜스포머 | O | O |

# **단어 위치 인코딩**

# 앞서 살펴 본 트랜스포머 인코더는 셀프 어텐션과 밀집층을 사용하기에
# 단어순서를 제대로 활용하지는 못한다.
# 하지만 단어 인코딩 과정에서 단어순서 정보를 활용하도록 하는 기능을 추가하면
# 트랜스포머가 알아서 단어위치 정보를 활용한다.
# 
# 다음 `PositionalEmbedding` 층 클래스는 두 개의 임베딩 클래스를 사용한다.
# 하나는 보통의 단어 임베딩이며,
# 다른 하나는 단어의 위치 정보를 임베딩한다. 
# 각 임베딩의 출력값을 합친 값을 트랜스포머에게 전달하는 역할을 수행한다.

# ```python
# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.token_embeddings = layers.Embedding(
#             input_dim=input_dim, output_dim=output_dim)
#         self.position_embeddings = layers.Embedding(
#             input_dim=sequence_length, output_dim=output_dim)
#         self.sequence_length = sequence_length
#         self.input_dim = input_dim
#         self.output_dim = output_dim
# 
#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_tokens = self.token_embeddings(inputs)
#         embedded_positions = self.position_embeddings(positions)
#         return embedded_tokens + embedded_positions
# 
#     def compute_mask(self, inputs, mask=None):
#         return tf.math.not_equal(inputs, 0)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "output_dim": self.output_dim,
#             "sequence_length": self.sequence_length,
#             "input_dim": self.input_dim,
#         })
#         return config
# ```

# **단어위치 인식 트랜스포머 아키텍처**

# 아래 코드는 `PositionalEmbedding` 층을 활용하여 트랜스포머 인코더가 
# 단어위치를 활용할 수 있도록 한다. 
# 최종 모델의 테스트셋에 대한 정확도가 88.3%까지 향상됨을 확인할 수 있다.

# ```python
# vocab_size = 20000
# sequence_length = 600
# embed_dim = 256
# num_heads = 2
# dense_dim = 32
# 
# inputs = keras.Input(shape=(None,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# ```

# ### 트랜스포머 사용 기준

# 단어주머니 모델이 여전히 유용하게 활용된다.
# 실제로 IMDB 데이터셋에 대한 성능도 단어주머니 모델이 가장 좋았다.
# 그리고 많은 실험 결과 "훈련셋의 크기"와 "텍스트의 평균 단어 수"의 비율이 모델 선택에
# 결정적인 역할을 수행한다는 경험 법칙이 알려졌다.
# 
# - ("훈련셋의 크기" $/$ "텍스트의 평균 단어 수") $>$ 1500 인 경우: 트랜스포머 등 순차 모델
# - ("훈련셋의 크기" $/$ "텍스트의 평균 단어 수") $<$ 1500 인 경우: 바이그램 단어주머니 모델

# **예제 1**

# 1천개의 단어를 포함한 텍스트 십만 개로 이루어진 훈련셋을 사용하는 경우 비율이 100이기에 바이그램 모델을 사용하는 것이 좋다.

# **예제 2**

# 평균 40개의 단어를 포함하는 트윗(tweets) 5만 개로 이루어진 훈련셋을 사용하는 경우 비율이 1,250이기에 역시 바이그램 모델을 사용하는 것이 좋다.

# **예제 3**

# 평균 40개의 단어를 포함하는 트윗(tweets) 50만 개로 이루어진 훈련셋을 사용하는 경우 비율이 12,500이기에 이번엔 트랜스포머 인코더를 활용하는 것이 좋다.

# **예제 4**

# IMDB 훈련셋은 2만 개의 후기로 구성되며 후기 한 개는 평균 233 개의 단어를 포함한다. 비율이 85.84 정도이기에 바이그램 모델이 보다 적합해야 하는데
# 지금까지 살펴본 결과가 이에 입증한다.

# **경험 법칙의 직관적 이해**

# 짧은 문장이 많을 수록 문맥을 파악하려면 단어들의 순서가 중요하며,
# 문장에 사용된 단어들 사이의 복잡한 연관성을 보다 주의깊게 살펴볼 필요가 있다.
# 예를 들어, "그 영화는 실패야"와 "그 영화는 실패였어"는 분명 다른 의미를 가지지만
# 단어주머니 모델은 차이점을 파악하기 어렵다. 
# 반면에 보다 긴 문장의 주제와 긍정/부정 등의 감성에 대한 분류는
# 단어 관련 통계의 중요성이 보다 크다.

# **주의사항** 
# 
# 앞서 설명한 경험 법칙은은 텍스트 분류에 한정된다. 
# 예를 들어 기계 번역<font size='2'>machine translation</font>의 경우 매우 긴 문장을 다룰 때 
# 트랜스포머가 가본적으로 가장 강력한 성능의 모델을 생성한다. 
