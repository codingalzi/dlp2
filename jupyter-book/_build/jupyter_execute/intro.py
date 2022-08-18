#!/usr/bin/env python
# coding: utf-8

# **소개**
# 
# 머신러닝/딥러닝 기술이 획기적으로 발전하면서 데이터 분석 및 인공지능 관련 연구의 
# 중요성이 자율주행, 의료영상, 농업, 교육, 재난 예방, 제조업 등 
# 사회, 경제, 산업의 거의 모든 분야에 지대한 영향을 미치고 있다.
# 이런 경향이 앞으로 더욱 강화될 것으로 기대되며 약간의 코딩 지식을 가진
# 어느 누구나 데이터로부터 학습하는 스마트 앱을 개발 할 수 있을 정도로
# 딥러닝 기술이 누구나 쉽게 활용할 수 있을 정도로 대중화될 것이다.
# 
# 여기서는 딥러닝의 기본 개념과 함께 다양한 딥러닝 기법을 최대한 직관적으로 전달하는 일에
# 집중하며, 이를 위해 다양한 예제와 적절한 코드를 이용한다.
# 예제에 사용되는 코드는 텐서플로우<font size='2'>TensorFlow</font> 2와
# 케라스<font size='2'>Keras</font>를 파이썬 딥러닝 프레임워크로 사용한다.
# 
# 

# **주요 내용**
# 
# 여기서 소개하는 내용은 프랑소와 숄레의 
# [Deep Learning with Python(2판)](https://www.manning.com/books/deep-learning-with-python-second-edition)의
# [주피터 노트북](https://github.com/fchollet/deep-learning-with-python-notebooks) 내용을 
# 기본 참고서로 사용하며 아래 내용을 전달한다.
# 
# - 딥러닝 개념
# - 딥러닝 활용
#     - 컴퓨터비전: 이미지 분석 및 분할
#     - 자연어처리: 시계열 예측, 텍스트 분류, 자동 번역, 텍스트 생성

# **전제 사항**
# 
# 코드 이해를 위해 파이썬 프로그래밍과 넘파이<font size='2'>NumPy</font>에 대한 
# 기초 이상의 지식이 요구된다.
# 또한 머신러닝에 대한 기초 지식이 있는 경우 보다 쉽게 딥러닝을 이해할 수 있다.
# 
# 아래 강의노트를 활용할 것을 추천한다.
# 
# - [파이썬 왕기초](https://hj617kim.github.io/core_pybook/)
# - [파이썬 데이터 분석](https://codingalzi.github.io/datapy/)
# - [핸즈온 머신러닝(3판)](https://codingalzi.github.io/handson-ml3/)
