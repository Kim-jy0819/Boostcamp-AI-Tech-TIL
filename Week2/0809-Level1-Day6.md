# 부스트캠프 6일차

- [부스트캠프 6일차](#부스트캠프-6일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[1강] 딥러닝 기본 용어 설명](#1강-딥러닝-기본-용어-설명)
    - [[2강] 뉴럴 네트워크 - MLP (Multi-Layer Perceptron)](#2강-뉴럴-네트워크---mlp-multi-layer-perceptron)
  - [피어 세션 정리](#피어-세션-정리)
    - [딥러닝 논문 리뷰 스터디 시작](#딥러닝-논문-리뷰-스터디-시작)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/9 (월)
  - [x] DL Basic
    - [x] (1강) 딥러닝 기본 용어 설명 - Historical review
    - [x] (2강) 뉴럴 네트워크 - MLP
    - [x] [필수 과제1] MLP Assignment
  - [x] 논문 발표 준비하기

## 강의 내용 정리

### [1강] 딥러닝 기본 용어 설명

* 딥러닝 중요 요소 : data, model, loss function, optimization algorithm 등
  * loss function : 이루고자 하는 것의 근사치
    * ![image](https://user-images.githubusercontent.com/35680202/128652035-6e7f92a1-2929-4e2f-9b05-0873aefdc5d8.png)
* [Historical Review](https://dennybritz.com/blog/deep-learning-most-important-ideas/)
  * 2012 - AlexNet : ImageNet challenge 에서 딥러닝 기법으로 처음 1등
  * 2013 - DQN : 강화학습 Q러닝, 딥마인드
  * 2014 - Encoder/Decoder : NMT(Neural Machine Translation) 기계어번역
  * 2014 - Adam Optimizer : 웬만하면 학습이 잘 된다.
  * 2015 - GAN : 네트워크가 generator와 discriminator 두개를 만들어서 학습
  * 2015 - Residual Networks : 네트워크를 깊게 쌓을 수 있게 만들어줌
  * 2017 - Transformer (Attention Is All You Need) : 기존 방법론들을 대체할 정도의 영향력
  * 2018 - BERT (Bidirectional Encoder Representations from Transformers) : 'fine-tuned' NLP models 발전 시작
  * 2019 - BIG Language Models : fine-tuned NLP model의 끝판왕, OpenAI GPT-3
  * 2020 - Self-Supervised Learning : SimCLR (a simple framework for contrastive learning of visual representations)

### [2강] 뉴럴 네트워크 - MLP (Multi-Layer Perceptron)

* Neural networks are **function approximators** that stack affine tansformations  followed by nonlinear transformations : 행렬 곱과 비선형 연산이 반복되면서, 함수를 근사하는 모델
* Linear Neural Networks
  * <!-- $y = W^Tx + b$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20W%5ETx%20%2B%20b">에서 <!-- $W$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W">를 찾는 것은 서로 다른 두 차원에서의 선형변환을 찾겠다는 것
* Multi-Layer Perceptron
  * <!-- $y = W_2^Th = W_2^T W_1^T x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20W_2%5ETh%20%3D%20W_2%5ET%20W_1%5ET%20x">​ 는 linear neural network와 다를 바가 없다.
  * <!-- $y = W_2^Th = W_2^T \rho(W_1^T x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20W_2%5ETh%20%3D%20W_2%5ET%20%5Crho(W_1%5ET%20x)">​ 와 같은 Nonlinear transform이 필요하다.
  * Multilayer Feedforward Networks are Universal Approximators : 뉴럴 네트워크의 표현력이 그만큼 크다. 하지만 어떻게 찾을지는 (알아서해)
* [PyTorch official docs](https://pytorch.org/docs/stable/nn.html)

## 피어 세션 정리

### 딥러닝 논문 리뷰 스터디 시작

* 목차
  1. [VGG-16](https://arxiv.org/abs/1409.1556) (2014)
  2. [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) (2015) 🙋‍♀️
  3. [GoogLeNet(Inception-v1)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) (2014)
  4. [ResNet-50](https://arxiv.org/abs/1512.03385) (2015)
  5. [Inception-v3](https://arxiv.org/abs/1512.00567v3) (2015)
  6. [XceptionNet](https://arxiv.org/abs/1610.02357) (2016)
  7. [DenseNet](https://arxiv.org/abs/1608.06993) (2017)
  8. [Inception-v4, Inception-ResNet](https://arxiv.org/abs/1602.07261v2) (2016) 🙋‍♀️
  9. [ResNeXt-50](https://arxiv.org/abs/1611.05431) (2017)
  10. [EfficientNet](https://arxiv.org/abs/1905.11946) (2019)
  11. [EfficientNet v2](https://arxiv.org/pdf/2104.00298.pdf) (2020)
* 방법
  * 하루에 두 명씩 발표
  * 내용 : 제안 배경 / 모델 구조 / 실험 결과 등등
  * 레벨 : 구조에 대한 설명 충분히
  * 형식 : 발표자료는 자유롭게

## 학습 회고

* 논문 리뷰 스터디를 팀원들과 같이 하게 되었다. 혼자 읽을 생각에 막막했는데 다행이다.
* 한달 전에 읽은 적 있는 논문인데도 지금 정리하는데 꽤 걸렸다. 역시 발표 준비하면서 읽는 거랑 혼자 읽는 거랑은 완전히 다른 것 같다.
* 수학공부, NLP공부는 언제 할지 다시 고민해봐야겠다.
