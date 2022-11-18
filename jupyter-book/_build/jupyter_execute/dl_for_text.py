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

# - 자연어처리(Natural Language Processing) 소개
#     - 단어주머니(bag-of-words) 모델
#     - 순차(sequence) 모델
# - 순차 모델 활용
#     - 양방향 순환신경망(bidirectional LSTM) 적용
# - 트랜스포머(Transformer) 활용
# - 시퀀스-투-시퀀스(seq2seq) 모델 활용

# ## 자연어처리 소개

# 파이썬, 자바, C, C++, C#, 자바스크립트 등 컴퓨터 프로그래밍언어와 구분하기 위해 
# 일상에서 사용되는 한국어, 영어 등을 **자연어**<font size='2'>natural language</font>라 
# 부른다. 
# 
# 자연어의 특성상 정확한 분석을 위한 알고리즘을 구현하는 일은 사실상 매우 어렵다. 
# 딥러닝 기법이 활용되기 이전까지는 적절한 규칙을 구성하여 자연어를 이해하려는 
# 수 많은 시도가 있어왔지만 별로 성공적이지 않았다.

# 1990년대부터 인터넷으로부터 구해진 엄청난 양의 텍스트 데이터에 머신러닝 기법을
# 적용하기 시작했다. 단, 주요 목적이 **언어의 이해**가 아니라 
# 아래 예제들처럼 입력 텍스트를 분석하여 
# **통계적으로 유용한 정보를 예측**하는 방향으로 수정되었다.
# 
# - 텍스트 분류: "이 문장의 주제는?"
# - 내용 필터링: "욕설이 포함되었나?"
# - 감성 분석: "내용이 긍정이야 부정이야?"
# - 언어 모델링: "이 문장에 이어 어떤 단어가 있어야 하지?"
# - 번역: "이거를 한국어로 어떻게 말해?"
# - 요약: "이 기사를 한 줄로 요약하면?"
# 
# 이와 같은 분석을 **자연어 처리**<font size='2'>Natural Language Processing</font>이라 하며
# 단어, 문장, 문단 등에서 찾을 수 있는 패턴을  인식하려 시도한다. 

# **머신러닝 활용**

# 자연어처리를 위해 1990년대부터 시작된 머신러닝 활용의 변화과정은 다음과 같다.

# - 1990 - 2010년대 초반: 
#     결정트리(decision trees), 로지스틱 회귀(logistic regression) 모델이 주로 활용됨.
# 
# - 2014-2015: LSTM 등 시퀀스 처리 알고리즘 활용 시작
# 
# - 2015-2017: (양방향) 순환신경망이 기본적으로 활용됨.
# 
# - 2017-2018: 트랜스포머<font size='2'> transformer</font> 모델이 최고의 성능 발휘하며, 
#     많은 난제들을 해결함. 현재 가장 많이 활용되는 모델임.

# ## 텍스트 벡터화

# 딥러닝 모델은 텍스트 자체를 처리할 수 없다.
# 따라서 텍스트를 수치형 텐서로 변환하는 **텍스트 벡터화**<font size='2'>text vectorization</font> 과정이 요구되며
# 보통 다음 세 단계를 따른다.
# 
# 1. **텍스트 표준화**<font size='2'>text standardization</font>: 소문자화, 마침표 제거 등등
# 1. **토큰화**<font size='2'>tokenization</font>: 기본 단위의 **유닛**<font size='2'>units</font>으로 쪼개기.
#     문자, 단어, 단어 집합 등이 토큰으로 활용됨.
# 1. **어휘 색인화**<font size='2'>vocabulary indexing</font>: 토큰 각각을 하나의 수치형 벡터로 변환.
# 
# 아래 그림은 텍스트 벡터화의 기본적인 과정을 잘 보여준다.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-01.png" style="width:60%;"></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.manning.com/books/deep-learning-with-python-second-edition">Deep Learning with Python(2판)</a>&gt;</div></p>

# ### 텍스트 표준화

# 다음 두 문장을 표준화를 통해 동일한 문장으로 변환해보자.
# 
# - "sunset came. i was staring at the Mexico sky. Isnt nature splendid??"
# - "Sunset came; I stared at the M&eacute;xico sky. Isn't nature splendid?"

# 예를 들어 다음 표준화 기법을 사용할 수 있다.
# 
# - 모두 소문자화
# - `.`, `;`, `?`, `'` 등 특수 기호 제거
# - 특수 알파벳 변환: "&eacute;"를 "e"로, "&aelig;"를 "ae"로 등등
# - 동사/명사의 기본형 활용: "cats"를 "[cat]"로, "was staring"과 "stared"를 "[stare]"로 등등.

# 그러면 위 두 문장 모두 아래 문장으로 변환된다.

# - "sunset came i [stare] at the mexico sky isnt nature splendid"

# 표준화 과정을 통해 어느 정도의 정보를 상실하게 되지만
# 학습해야할 내용을 줄여 일반화 성능이 보다 좋은 모델을 훈련시키는 장점이 있다.
# 하지만 분석 목적에 따라 표준화 기법은 경우에 따라 달라질 수 있음에 주의해야 한다. 
# 예를 들어 인터뷰 기사의 경우 물음표(`?`)는 제거하면 안된다.

# ### 토큰화

# 텍스트 표준화 이후 데이터 분석의 기본 단위인 토큰으로 쪼개야 한다.
# 보통 아래 세 가지 방식 중에 하나를 사용한다.
# 
# - 단어 기준 토큰화(word-level tokenization)
#     - 공백으로 구분된 단어들로 쪼개기. 
#     - 경우에 따라 동사 어근과 어미를 구분하기도 함: "star+ing", "call+ed" 등등
# - N-그램 토큰화(N-gram tokenization)
#     - N-그램 토큰: 연속으로 위치한 N 개(이하)의 단어 묶음
#     - 예제: "the cat", "he was" 등은 2-그램 토큰이다.
# - 문자 기준 토큰화(character-level tokenization)
#     - 하나의 문자가 하나의 토큰임.
#     - 문장 생성, 음성 인식 등에서 활용됨.

# 일반적으로 문자 기준 토큰화는 잘 사용되지 않는다. 
# 여기서도 단어 기준과 N-그램 토큰화만 이용한다.
# 
# - 단어 기준 토큰화: 단어들의 순서를 중요시하는 **순차 모델**<font size='2'>sequence models</font>을 사용할 경우 활용
# - N-그램 토큰화: 단언들의 순서를 별로 상관하지 않는 **단어 주머니**<font size='2'>bag-of-words</font> 
#     모델을 사용할 경우 활용
#     - N-그램: 단어들 사이의 순서에 대한 지엽적 정보를 어느 정도 유지함.
#     - 일종의 특성 공학<font size='2'>feature engineering</font> 기법이며 따라서 
#         얕은 학습 기반의 언어처리 모델에 활용됨.
#     - 1차원 합성곱 신경망, 순환 신경망, 트랜스포머 등은 이 기법을 사용하지 않아도 됨.

# 단어주머니(bag-of-words)는 N-토큰으로 구성된 집합을 의미하며 
# **N-그램 주머니**라고도 불린다.
# 예를 들어 "the cat sat on the mat." 문장에 대한 
# 2-그램 집합과 3-그램 집합은 각각 다음과 같다.

# - 2-그램 집합
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
# 영화 후기에서 삭제하도록 하였다.
# 
# ```python
# from tensorflow.keras.datasets import imdb
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# ```

# 케라스의 imdb 데이터셋은 이미 정수들의 시퀀스로 전처리가 되어 있다. 
# 하지만 여기서는 원본 imdb 데이터셋을 대상으로 전처리를 직접 수행하는 단계부터 살펴볼 것이다.
# 이를 위해 아래 사항을 기억해 두자.
# 
# - OOV 인덱스 활용: 어휘 색인에 포함되지 않는 단어는 모두 1로 처리됨. 
#     일반 문장으로 재번역되는 경우 "[UNK]" 으로 처리됨.
#     - OOV = Out Of Vocabulary (미동록 어휘)
#     - UNK = Unknown (미확인)
# - 마스크 토큰<font size='2'>mask token</font>: 문장의 길이를 맞추기 위한 패딩으로 사용되는 0을
#     가리키며, 훈련 과정에서 무시됨.    
#     ```
#     [[5,  7, 124, 4, 89],
#      [8, 34,  21, 0,  0]]
#     ```

# ### `TextVectorization` 층 활용

# 케라스의 `TextVectorization` 층을 이용하여 텍스트 벡터화를 진행할 수 있다.
# 
# 아래 코드는 `TextVectorization` 층 구성에 사용되는 주요 기본 설정을 보여준다.
# 표준화와 토큰화 방식을 임의로 지정해서 활용할 수도 있지만 여기서는 자세히 다루지 않는다.
# 
# - 표준화: `standardize='lower_and_strip_punctuation'` (소문자화와 마침표 등 제거)
# - 토큰화: `split='whitespace'` (단어 기준 쪼개기), `ngrams=None` (n-그램 미사용)
# - 출력 모드: `output_mode="int"` (정수 인코딩)

# ```python
# from tensorflow.keras.layers import TextVectorization
# 
# text_vectorization = TextVectorization(
#     standardize='lower_and_strip_punctuation',
#     split='whitespace',
#     ngrams=None,
#     output_mode='int',
# )
# ```

# 예를 들어, 아래 데이터셋을 이용하여 텍스트 벡터화를 해보자.

# ```python
# dataset = [
#     "I write, erase, rewrite",
#     "Erase again, and then",
#     "A poppy blooms.",
# ]
# ```

# 어휘 색인화를 위해 먼저 `adapt()` 메서드를 이용하여 어휘 색인을 만든다.

# ```python
# >>> text_vectorization.adapt(dataset)
# ```

# 생성된 어휘 색인은 다음과 같다.

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

# 생성된 어휘 색인을 활용하여 새로운 문장을 벡터화 해보자.

# ```python
# >>> test_sentence = "I write, rewrite, and still rewrite again"
# >>> encoded_sentence = text_vectorization(test_sentence)
# >>> print(encoded_sentence)
# tf.Tensor([ 7  3  5  9  1  5 10], shape=(7,), dtype=int64)
# ```

# 벡터화된 텐서로부터 문장을 복원하면 표준화된 문장이 생성된다.

# ```python
# >>> inverse_vocab = dict(enumerate(vocabulary))
# >>> decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
# >>> print(decoded_sentence)
# i write rewrite and [UNK] rewrite again
# ```

# :::{admonition} `TextVectorization` 층 사용법
# :class: info
# 
# `TextVectorization` 층은 GPU 또는 TPU에서 지원되지 않는다.
# 따라서 모델 구성에 직접 사용하는 방식은 모델의 훈련을
# 늦출 수 있기에 권장되지 않는다.
# 여기서는 대신에 데이터셋 전처리를 모델 구성과 독립적으로 처리하는 방식을 이용한다.
# 
# 하지만 훈련이 완성된 모델을 실전에 배치할 경우 `TextVectorization` 층을
# 완성된 모델에 추가해서 사용하는 게 좋다.
# :::

# ## 문장 표현법: 집합 대 시퀀스

# 앞서 언급한 대로 자연어처리 모델에 따라 단어 모음을 다루는 방식이 다르다. 
# 
# - 단어주머니(bag-of-words) 모델
#     - 단어들의 순서를 무시. 단어 모음을 단어들의 집합으로 다룸.
#     - 2015년 이전까지 주로 사용됨.
# - 시퀀스(sequence) 모델
#     - RNN: 단어들의 순서를 시계열 데이터의 스텝처럼 간주. 2015-2016에 주로 사용됨.
#     - 트랜스포머<font size='2'>Transformer</font> 아키텍처. 
#         기본적으로 순서를 무시하지만 단어 위치를 학습할 수 있는 능력을 가짐.
#         2017년 이후 많이 활용됨.

# 여기서는 IMDB 영화 리뷰 데이터를 이용하여 두 모델 방식의 
# 활용법과 차이점을 소개한다.

# ### IMDB 영화 리뷰 데이터 준비

# 이전과는 달리 여기서는 IMDB 데이터셋을 직접 다운로드하여 전처리하는 
# 과정을 살펴본다. 

# 준비 과정 1: 데이터셋 다운로드 압축 풀기

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
# `train`의 `pos`와 `neg` 서브디렉토리에 각각 12,500개의 긍정과 부정 리뷰가
# 포함되어 있다. `aclImdb/train/unsup` 서브디렉토리는 필요 없기에 삭제한다.

# 긍정 리뷰 하나의 내용을 살펴보자.
# 모델 구성 이전에 훈련 데이터셋을 살펴 보고
# 모델에 대한 직관을 갖는 과정이 항상 필요하다.

# ```
# I first saw this back in the early 90s on UK TV, i did like it then but i missed the chance to tape it, many years passed but the film always stuck with me and i lost hope of seeing it TV again, the main thing that stuck with me was the end, the hole castle part really touched me, its easy to watch, has a great story, great music, the list goes on and on, its OK me saying how good it is but everyone will take there own best bits away with them once they have seen it, yes the animation is top notch and beautiful to watch, it does show its age in a very few parts but that has now become part of it beauty, i am so glad it has came out on DVD as it is one of my top 10 films of all time. Buy it or rent it just see it, best viewing is at night alone with drink and food in reach so you don't have to stop the film.<br /><br />Enjoy
# ```

# 준비 과정 2: 검증셋 준비

# 훈련셋의 20%를 검증셋으로 떼어낸다.
# 이를 위해 `aclImdb/val` 디렉토리를 생성한 후에
# 긍정과 부정 훈련셋 모두 무작위로 섞은 후 그중 20%를 검증셋 디렉토리로 옮긴다.

# 준비 과정 3: 텐서 데이터셋 준비

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
# 
# 배치의 크기는 32이며, 예를 들어, 첫째 배치의 입력과 타깃 데이터의 정보는 다음과 같다.

# ```python
# >>> for inputs, targets in train_ds:
# ...     # 예제: 첫째 배치의 첫째 리뷰
# ...     print("inputs[0]:", inputs[0])
# ...     print("targets[0]:", targets[0])    
# ...     break
# inputs[0]: tf.Tensor(b'The film begins with a bunch of kids in reform school and focuses on a kid named \'Gabe\', who has apparently worked hard to earn his parole. Gabe and his sister move to a new neighborhood to make a fresh start and soon Gabe meets up with the Dead End Kids. The Kids in this film are little punks, but they are much less antisocial than they\'d been in other previous films and down deep, they are well-meaning punks. However, in this neighborhood there are also some criminals who are perpetrating insurance fraud through arson and see Gabe as a convenient scapegoat--after all, he\'d been to reform school and no one would believe he was innocent once he was framed. So, when Gabe is about ready to be sent back to "The Big House", it\'s up to the rest of the gang to save him and expose the real crooks.<br /><br />The "Dead End Kids" appeared in several Warner Brothers films in the late 1930s and the films were generally very good (particularly ANGELS WITH DIRTY FACES). However, after the boys\' contracts expired, they went on to Monogram Studios and the films, to put it charitably, were very weak and formulaic--with Huntz Hall and Leo Gorcey being pretty much the whole show and the group being renamed "The Bowery Boys". Because ANGELS WASH THEIR FACES had the excellent writing and production values AND Hall and Gorcey were not constantly mugging for the camera, it\'s a pretty good film--and almost earns a score of 7 (it\'s REAL close). In fact, while this isn\'t a great film aesthetically, it\'s sure a lot of fun to watch, so I will give it a 7! Sure, it was a tad hokey-particularly towards the end when the kids take the law into their own hands and Reagan ignores the Bill of Rights--but it was also quite entertaining. The Dead End Kids are doing their best performances and Ronald Reagan and Ann Sheridan provided excellent support. Sure, this part of the film was illogical and impossible but somehow it was still funny and rather charming--so if you can suspend disbelief, it works well.', shape=(), dtype=string)
# targets[0]: tf.Tensor(1, shape=(), dtype=int32)
# ```

# ### 단어주머니 기법

# 단어주머니에 채울 토큰으로 어떤 N-그램을 사용할지 먼저 지정해야 한다. 
# 
# - 유니그램(unigrams): 하나의 단어가 한의 토큰
# - N-그램(N-grams): 최대 N 개의 연속 단어로 이루어진 토큰

# **방식 1: 유니그램 바이너리 인코딩**

# 예를 들어 "the cat sat on the mat" 문장을 유니그램으로 처리하면 다음 
# 단어주머니가 생성된다.
# 집합으로 처리되기에 단어들의 순서는 완전히 무시된다.
# 
# ```
# {"cat", "mat", "on", "sat", "the"}
# ```
# 
# 이제 모든 문장은 어휘색인에 포함된 단어들의 수만큼 긴 1차원 이진 텐서(binary tensor)로
# 처리된다. 
# 즉, 멀티-핫(multi-hot) 인코딩 방식을 사용해서 텐서로 변환된다.
# [4장](https://codingalzi.github.io/dlp/notebooks/dlp04_getting_started_with_neural_networks.html)과 
# [5장](https://codingalzi.github.io/dlp/notebooks/dlp05_fundamentals_of_ml.html)에서 
# 문장을 인코딩 방식과 동일하다. 
# 
# `TextVectorization` 클래스의 `output_mode="multi_hot"` 옵션을 이용하면
# 방금 설명한 내용을 그대로 처리해준다.

# ```python
# from tensorflow.keras.layers import TextVectorization
# 
# text_vectorization = TextVectorization(
#     max_tokens=20000,
#     output_mode="multi_hot",
# )
# 
# # 어휘색인 생성
# text_only_train_ds = train_ds.map(lambda x, y: x)
# text_vectorization.adapt(text_only_train_ds)
# ```

# 생성된 어휘색인을 이용하여 훈련셋, 검증셋, 테스트셋 모두 벡터화한다. 

# In[1]:


binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))


# 변환된 첫째 배치의 입력과 타깃 데이터의 정보는 다음과 같다.
# `max_tokens=20000`으로 지정하였기에 모든 문장은 길이가 2만인 벡터로 변환되었다.

# In[2]:


for inputs, targets in binary_1gram_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break


# **밀집 모델 지정**

# 단어주머니 모델로 여기서는 밀집 모델을 사용한다. 
# `get_model()` 함수가 컴파일 된 단순한 밀집 모델을 반환한다.
# 모델의 출력값은 긍정일 확률이며, 
# 최상위 층의 활성화 함수로 `sigmoid`를 사용한다.

# In[3]:


from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # 긍정일 확률 계산
    
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    return model


# In[4]:


model = get_model()
model.summary()


# **모델 훈련**

# 밀집 모델 훈련과정은 특별한 게 없다.
# 훈련 후 테스트셋에 대한 정확도가 89% 보다 조금 낮게 나온다.
# 최고 성능의 모델이 테스트셋에 대해 95% 정도 정확도를 내는 것보다는 낮지만
# 무작위로 찍는 모델보다는 훨씬 좋은 모델이다.

# In[5]:


callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
                                    save_best_only=True)
]

model.fit(binary_1gram_train_ds.cache(),
          validation_data=binary_1gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")


# **방식 2: 바이그램 바이너리 인코딩**

# 바이그램(2-grams)을 유니그램 대신 이용해보자. 
# 예를 들어 "the cat sat on the mat" 문장을 바이그램으로 처리하면 다음 
# 단어주머니가 생성된다.
# 
# ```
# {"the", "the cat", "cat", "cat sat", "sat",
#  "sat on", "on", "on the", "the mat", "mat"}
# ```
# 
# `TextVectorization` 클래스의 `ngrams=N` 옵션을 이용하면
# N-그램들로 이루어진 어휘색인을 생성할 수 있다.

# In[6]:


text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="multi_hot",
)


# 어휘색인 생성과 훈련셋, 검증셋, 테스트셋의 벡터화 과정은 동일하다. 

# In[7]:


text_vectorization.adapt(text_only_train_ds)

binary_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
binary_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
binary_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))


# 훈련 후 테스트셋에 대한 정확도가 90%를 조금 웃돌 정도로 많이 향상되었다.

# In[8]:


model = get_model()

callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras",
                                    save_best_only=True)
]

model.fit(binary_2gram_train_ds.cache(),
          validation_data=binary_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model("binary_2gram.keras")
print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")


# **방식 3: 바이그램 TF-IDF 인코딩**

# N-그램을 벡터화할 때 사용 빈도를 함께 저장하는 방식을 사용할 수 있다.
# 단어의 사용 빈도가 아무래도 문장 평가에 중요한 역할을 수행할 것이기 때문이다.
# 아래 코드에서처럼 `output_mode="count"` 옵션을 사용하면 된다.

# In[9]:


text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="count"
)


# 그런데 이렇게 하면 "the", "a", "is", "are" 등의 사용 빈도는 매우 높은 반면에
# "Chollet" 등의 단어는 빈도가 거의 0에 가깝게 나온다.
# 또한 생성된 벡터의 대부분은 0으로 채워질 것이다. 
# `max_tokens=20000`을 사용한 반면에 하나의 문장엔 많아야 몇 십개 정도의 단어만 사용되었기 때문이다. 
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
#     - 예를 들어, 하나의 리뷰에 "terrible" 이 많이 사용되었다면
#         해당 리뷰는 부정일 가능성 높음.
# - `IDF`(Inverse Document Frequency)
#     - 데이터셋 전체 문장에서 사용된 단어의 빈도
#     - 낮을 수록 중요. 
#     - "the", "a", "is" 등의 `IDF` 값은 높지만 별로 중요하지 않음.
# - `TF-IDF = TF / IDF`

# `output_mode="tf_idf"` 옵션을 사용하면 TF-IDF 인코딩을 지원한다.

# In[10]:


text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="tf_idf",
)


# 훈련 후 테스트셋에 대한 정확도가 다시 89% 아래로 내려간다.
# 여기서는 별 도움이 되지 않았지만 많은 텍스트 분류 모델에서는 1% 정도의 성능 향상을 가져온다.
# 
# **주의사항**: 아래 코드는 현재(Tensorflow 2.6과 2.7) GPU를 사용하지 않는 경우에만 작동한다. 
# 이유는 아직 모른다([여기 참조](https://github.com/fchollet/deep-learning-with-python-notebooks/issues/190)).

# In[11]:


text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
tfidf_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
tfidf_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
                                    save_best_only=True)
]
model.fit(tfidf_2gram_train_ds.cache(),
          validation_data=tfidf_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model("tfidf_2gram.keras")
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")


# **부록: 문자열 벡터화 전처리를 함께 처리하는 모델 내보내기**

# 훈련된 모델을 실전에 배치하려면 텍스트 벡터화도 모델과 함께 내보내야 한다.
# 이를 위해 `TextVectorization` 층의 결과를 재활용만 하면 된다.

# In[12]:


inputs = keras.Input(shape=(1,), dtype="string")
# 텍스트 벡터화 추가
processed_inputs = text_vectorization(inputs)
# 훈련된 모델에 적용
outputs = model(processed_inputs)

# 최종 모델
inference_model = keras.Model(inputs, outputs)


# `inference_model`은 일반 텍스트 문장을 직접 인자로 받을 수 있다.
# 예를 들어 "That was an excellent movie, I loved it."라는 리뷰는
# 긍정일 확률이 매우 높다고 예측된다.

# In[13]:


import tensorflow as tf

raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie, I loved it."],
])

predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")


# ### 시퀀스 모델 기법

# 앞서 살펴본 대로 바이그램(bigrams) 등을 이용하여 단어들 사이의 순서 정보를 함께 활용하면 기본적으로 훈련된 모델의 성능이 향상된다.
# 하지만 N-그램 등은 일종의 수동으로 진행하는 일종의 특성공학(feature engineering)이며,
# 딥러닝은 그런 특성공학을 가능하면 진행하지 않는 방향으로 발전해왔다.
# 여기서는 단언들의 순서를 그대로 함께 전달만 하고 나머지 특성은 모델 스스로 찾아내도록 하는 시퀀스 모델의 활용법을 살펴본다. 

# **IMDB 데이터셋 다운로드 및 준비**
# 
# [1부](https://codingalzi.github.io/dlp/notebooks/dlp11_part01_introduction.html)와 동일하다.

# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz
# 
# if 'google.colab' in str(get_ipython()):
#     !rm -r aclImdb/train/unsup
# else: 
#     import shutil
#     unsup_path = './aclImdb/train/unsup'
#     shutil.rmtree(unsup_path)

# import os, pathlib, shutil, random
# from tensorflow import keras
# 
# batch_size = 32
# base_dir = pathlib.Path("aclImdb")
# val_dir = base_dir / "val"
# train_dir = base_dir / "train"
# 
# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname,
#                     val_dir / category / fname)
# 
# train_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/train", batch_size=batch_size
#     )
# val_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/val", batch_size=batch_size
#     )
# test_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/test", batch_size=batch_size
#     )
# text_only_train_ds = train_ds.map(lambda x, y: x)

# **정수 시퀀스 데이터셋 준비**

# 훈련셋의 모든 리뷰 문장을 정수들의 벡터로 변환한다.
# 단, 리뷰 문장이 최대 600개의 단어만 포함하도록 한다. 
# 또한 사용되는 어휘는 빈도 기준 최대 2만개로 제한한다. 
# 
# - `max_length = 600`
# - `max_tokens = 20000`
# - `output_sequence_length=max_length`

# from tensorflow.keras import layers
# 
# max_length = 600
# max_tokens = 20000
# 
# text_vectorization = layers.TextVectorization(
#     max_tokens=max_tokens,
#     output_mode="int",
#     output_sequence_length=max_length,
# )
# 
# text_vectorization.adapt(text_only_train_ds)
# 
# int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
# int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
# int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))

# 변환된 첫째 배치의 입력과 타깃 데이터의 정보는 다음과 같다.
# `output_sequence_length=600`으로 지정하였기에 모든 문장은 단어를 최대 600개에서
# 잘린다. 따라서 생성되는 정수들의 벡터는 길이가 모두 600으로 지정된다.
# 물론 문장이 600개보다 적은 수의 단어를 사용한다면 나머지는 0으로 채워진다. 
# 또한 벡터에 사용된 정수는 2만보다 작은 값이며, 
# 이는 빈도가 가장 높은 2만개의 단어만을 대상(`max_tokens=20000`)으로 했기 때문이다.
# 
# 리뷰 문장의 길이를 600개의 단어로 제한한 이유는 리뷰가 평균적으로 233개의 단어를 사용하기 때문이다.
# 그리고 600 단어 이상을 사용하는 리뷰는 전체의 5% 정도에 불과하다.

# In[14]:


for inputs, targets in int_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break


# **벡터 원-핫 인코딩**

# 아래에서 소개하는 시퀀스 모델은 정수들의 벡터를 원-핫 인코딩된 벡터들의 시퀀스로 변환해서 사용한다.
# 예를 들어 `[2, 1, 4, 0, 0]` 벡터를 원-핫 인코딩하면 아래 결과를 얻는다.
# 단, 벡터에 사용된 정수는 0에서 4까지라고 가정한다. 
# 
# ```python
# [[0, 0, 1, 0, 0],
#  [0, 1, 0, 0, 0],
#  [0, 0, 0, 0, 1],
#  [1, 0, 0, 0, 0],
#  [1, 0, 0, 0, 0]]
# ```
# 
# `tf.one_hot()` 함수가 원-핫 인코딩을 실행한다. 
# 에를 들어 위 결과는 아래 방식으로 얻어진다.
# 
# ```python
# tf.one_hot(indices=[2, 1, 4, 0, 0], depth=5)
# ```

# **시퀀스 모델 예제 1**
# 
# - 원-핫 인코딩 활용: 입력값을 바로 원-핫 인코딩함.
# - 양방향 LSTM 모델 활용
#     - 1차원 합성곱 신경망도 경우에 따라 유사한 성능을 발휘하지만 거의 사용되지 않음.

# In[15]:


import tensorflow as tf

inputs = keras.Input(shape=(None,), dtype="int64")

# 원-핫 인코딩
embedded = tf.one_hot(inputs, depth=max_tokens)  # (600, 20000) 모양의 출력값 생성

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()


# **모델 훈련**

# 모델 훈련이 매우 느리다. 
# 이유는 입력 데이터가 너무 많은 특성을 갖기 때문이다. 
# 입력 데이터 하나의 모양과 특성 수는 다음과 같다.
# 
# - 모양: `(600, 20000)`
# - 특성 수: `600 * 20,000 = 12,000,000`
# 
# 양방향 LSTM은 엄청난 양의 반복을 실행하기에 당연히 훈련 시간이 길어진다.
# 게다가 훈련된 모델의 성능이 별로 좋지 않다.
# 테스테셋에 대한 정확도가 87% 정도에 불과해서
# 바이그램 모델보다 성능이 낮다.

# **주의사항**: 모델 훈련과정을 한 번 보기만 하려면 `epochs=1`로 설정하는 것을 권장한다.
# 책에서는 원래 `epochs=10`을 사용하였는데 컴퓨터 성능에 따라 몇 시간이 소요될 수 있다.

# In[16]:


callbacks = [
    keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=1, callbacks=callbacks)

model = keras.models.load_model("one_hot_bidir_lstm.keras")

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# **시퀀스 모델 예제 2: 단어 임베딩 활용**

# 앞서 보았듯이 원-핫 인코딩은 별로 적절하지 않다. 
# 원-핫 인코딩은 단어들의 순서는 잘 반영하지만 단어들 사이의 관계는 전혀 반형하지 못한다.
# 
# - "movie"와 "film", "비디오"와 "동영상", "강아지"와 "개" 등이 사실상 동일하다는 사실
# - "왕"(남자)과 "여왕"(여자), "boy"와 "girl" 등의 성별 관계
# - "king"의 복수는 "kings" 등 문법 관계
# - "고양이"와 "호랑이"는 고양이과, "개"와 "늑대"는 개과, "고양이"와 "개"는 애완동물, "늑대"와 "호랑이"는 야생동물 등의 관계

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-03.png" style="width:20%;"></div>
# 
# 그림 출처: [Deep Learning with Python(Manning MEAP)](https://www.manning.com/books/deep-learning-with-python-second-edition)

# 반면에 **단어 임베딩**(word embedding)은 단어들 사이의 관계를 모델 스스로 학습과정에서 찾도록 유도한다.
# 단어 임베딩을 활용하는 방법은 일반적으로 다음 두 가지이다.
# 
# - 모델 훈련과 동시에 단어 임베딩 학습도 진행하는 방식
#     - 자연어 종류와 모델 훈련 목적에 따라 기본적으로 서로 다른 단어 사이의 관계가 학습되어야 함.
#     - 예를 들어, 영화 리뷰 분석과 재판 판결문을 분석할 때 사용되는 단어 임베딩은 많이 다름.
# - 기존에 잘 훈련된 워드 임베딩 활용 방식

# **케라스의 `Embedding` 층 활용**

# 케라스의 `Embedding` 층은 일종의 사전처럼 작동한다. 
# 하나의 문장에 해당하는 정수들의 벡터가 입력값으로 들어오면 단어들간에 존재하는 연관성을 (어떤식으로라도) 담은 
# 부동소수점들의 벡터로 이루어진 시퀀스를 반환한다.
# 아래 그림은 원-핫 인코딩 방식과 단어 임베딩 방식의 차이점을 보여준다. 
# 
# - 원-핫 인코딩: 특성 수가 너무 많음
# - 단어 임베딩: 단어들 사이의 연관성을 256개, 512개, 1024개 정도 수준에서 찾음.

# <div align="center"><img src="https://drek4537l1klr.cloudfront.net/chollet2/Figures/11-02.png" style="width:45%;"></div>
# 
# 그림 출처: [Deep Learning with Python(Manning MEAP)](https://www.manning.com/books/deep-learning-with-python-second-edition)

# 예를 들어, 600 단어로 이루어진 문장을 단어 임베딩할 때 무엇인지 모르지만 단어들 사이의 연관성을 256개 찾으라 하면
# `(600, 256)` 모양의 텐서(단어 벡터)를 생성한다. 
# 즉, 600개의 단어 각각이 총 2만개의 어휘 색인에 포함된 단어들과의 연관성을 256개 찾는다.
# 
# 방금 설명한 것을 아래 코드가 실행한다. 
# 
# ```python
# layers.Embedding(input_dim=20000, output_dim=256)
# ```
# 
# 아래 코드는 단어 임베딩을 모델 구성에 직접 활용하는 것을 보여준다.
# 여전히 양방향 LSTM 층을 사용한다.

# In[17]:


inputs = keras.Input(shape=(None,), dtype="int64")

# 단어 임베딩
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("embeddings_bidir_gru.keras")

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# 훈련은 원-핫 인코딩 방식보다 훨씬 빠르게 이루어지며 성능은 87% 정도로 비슷하다. 
# 바이그램 모델보다 성능이 여전히 떨어지는 이유 중에 하나는 리뷰에 사용된 단어의 수를 600개로 제한하였기 때문이다. 

# **패딩(padding)과 마스킹(masking)**

# 반면에 리뷰 문장의 길이가 600이 되지 않는 경우 나머지는 **패딩**(padding)에 의해 0으로 채워진다.
# 하지만 이렇게 의미 없이 추가된 0이 훈련에 좋지 않은 영향을 미친다.
# 따라서 모델이 패딩을 위해 차가된 0이 있다는 사실을 인식하도록 도와주는 **마스킹**(masking)
# 기능을 활용하면 좋다.
# 
# 아래 코드는 마스킹을 활용하는 방식을 보여준다.
# 
# - `mask_zero=True` 옵션: 마스킹 옵션 켜기

# In[18]:


inputs = keras.Input(shape=(None,), dtype="int64")

# 마스킹 활용 단어 임베딩
embedded = layers.Embedding(
    input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# 모델 성능이 88% 정도로 살짝 향상된다.

# **훈련된 단어 임베딩 활용**

# 합성곱 신경망에서 이미지넷 등의 대용량 데이터셋을 활용하여 잘 훈련된 모델을 재활용하였던 것처럼
# 잘 구성된 대용량의 어휘 색인을 활용할 수 있다.
# 여기서는 수 백만 개의 단어를 활용하여 생성된 2014년에 스탠포드 대학교의 연구자들이 생성한
# [GloVe(Gloval Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/) 단어 임베딩을 활용한다.

# - GloVe 단어 임베딩 다운로드

# if 'google.colab' in str(get_ipython()):
#     !wget http://nlp.stanford.edu/data/glove.6B.zip
#     !unzip -q glove.6B.zip
# else: 
#     try: 
#         import wget, zipfile
#     except ModuleNotFoundError: 
#         !pip install wget
#         
#     import wget, zipfile
#     wget.download('http://nlp.stanford.edu/data/glove.6B.zip')
#     with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
#         zip_ref.extractall('./')

# - GloVe 워드 임베딩 파일 파싱

# import numpy as np
# path_to_glove_file = "glove.6B.100d.txt"
# 
# embeddings_index = {}
# 
# with open(path_to_glove_file) as f:
#     for line in f:
#         word, coefs = line.split(maxsplit=1)
#         coefs = np.fromstring(coefs, "f", sep=" ")
#         embeddings_index[word] = coefs
# 
# print(f"Found {len(embeddings_index)} word vectors.")

# - GloVe 단어 임베딩 행렬 준비

# embedding_dim = 100
# 
# vocabulary = text_vectorization.get_vocabulary()
# word_index = dict(zip(vocabulary, range(len(vocabulary))))
# 
# embedding_matrix = np.zeros((max_tokens, embedding_dim))
# for word, i in word_index.items():
#     if i < max_tokens:
#         embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# - 임베딩 층 준비

# embedding_layer = layers.Embedding(
#     max_tokens,
#     embedding_dim,
#     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#     trainable=False,
#     mask_zero=True,
# )

# - GloVe 임베딩 활용 모델 구성 및 훈련

# ```python
# inputs = keras.Input(shape=(None,), dtype="int64")
# 
# # GloVe 단어 임베딩 활용
# embedded = embedding_layer(inputs)
# 
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
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
# 
# callbacks = [
#     keras.callbacks.ModelCheckpoint("glove_embeddings_sequence_model.keras",
#                                     save_best_only=True)
# ]
# 
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
# 
# model = keras.models.load_model("glove_embeddings_sequence_model.keras")
# 
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
# ```
