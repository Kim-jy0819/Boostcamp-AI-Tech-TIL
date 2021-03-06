# 부스트캠프 7일차

- [부스트캠프 7일차](#부스트캠프-7일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[3강] Optimization](#3강-optimization)
  - [과제 수행 과정](#과제-수행-과정)
  - [도메인 특강](#도메인-특강)
  - [피어 세션 정리](#피어-세션-정리)
  - [마스터 클래스](#마스터-클래스)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/10 (화)
  - [x] DL Basic
    - [x] (3강) Optimization
    - [x] [필수 과제2] Optimization Assignment
  - [x] 도메인 특강 15:30~16:30 서대원, 박선규 (Upstage) - NLP, CV 도메인의 특징과 각 도메인 별 현업에서의 경험
  - [x] 피어세션 논문 리뷰 스터디 발표 - Batch Normalization
  - [x] 마스터클래스 18:00~19:00 안수빈 마스터님 - 데이터 리터러시와 시각화

## 강의 내용 정리

### [3강] Optimization

* **Generalization** : 일반화 성능을 높이는 것이 우리의 목표
  * Generalization gap : Training error와 Test error의 차이
  * **(k-fold) Cross-validation** : 최적의 하이퍼파라미터 찾기

* **Bias and Variance Tradeoff**
  * Bias : 평균적으로 봤을 때 정답에 가까우면 bias가 낮음
  * Variance : 출력이 일관적이면 variance가 낮음
  * Tradeoff
    * minimizing cost = (bias^2, variance, noise) 를 낮추는 것
    * 하나가 낮으면 하나가 높을 수 밖에 없다
    * ![image](https://user-images.githubusercontent.com/35680202/128812048-311e7b47-9543-4c0c-b5ce-1b7e2ba9e725.png)

* **Bootstrapping** : test하거나 metric을 계산하기 전에 random sampling하는 것
  * **Bagging**(Bootstrapping aggregating)
    * bootstrapping을 이용해서 여러 모델을 학습시킨 후 결과를 합치겠다.(voting or averaging)
    * 모든 모델이 독립적으로 돌아감
  * **Boosting**
    * 하나하나의 모델들을 시퀀셜하게 합쳐서 하나의 모델을 만든다.
    * 이전 모델이 잘 예측하지 못한 부분을 보완하기 위한 방식으로 학습해나감

* **Gradient Descent** : 1차 미분 이용, local minimum을 찾는 알고리즘
  * Stochastic gradient descent : 엄밀히 말하면 SGD는 한개의 샘플로 업데이트 하는 것
  * Mini-batch gradient descent
  * Batch gradient descent

* Batch-size Matters : 올바른 배치 사이즈는?
  * 배치 사이즈를 작게 쓰면 Flat minimizer에 수렴 : generalization performance가 더 높다.
  * 배치 사이즈를 크게 쓰면 Sharp minimizer에 수렴
  * ![image](https://user-images.githubusercontent.com/35680202/128814469-2ec789a8-e130-4c2b-8cfa-744e1e990685.png)

* Gradient Descent Methods
  * **(Stochastic) gradient descent** : 적절한 learning rate를 넣는 것이 중요
    * <!-- $W_{t+1} \leftarrow W_t - \eta g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%5Cleftarrow%20W_t%20-%20%5Ceta%20g_t">​
      * <!-- $\eta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ceta"> : Learning rate
      * <!-- $g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=g_t"> : Gradient
  * **Momentum** : 관성, 현재 gradient를 가지고 momentum을 accumulation 한다. 한번 흘러간 gradient direction을 어느정도 유지시켜주기 때문에 gradient가 왔다갔다해도 어느정도 잘 학습된다.
    * <!-- $a_{t+1} \leftarrow \beta a_t + g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a_%7Bt%2B1%7D%20%5Cleftarrow%20%5Cbeta%20a_t%20%2B%20g_t">
      * <!-- $\beta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta"> : momentum
      * <!-- $a_{t+1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a_%7Bt%2B1%7D"> : accumulation
    * <!-- $W_{t+1} \leftarrow W_t - \eta a_{t+1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%5Cleftarrow%20W_t%20-%20%5Ceta%20a_%7Bt%2B1%7D">
  * **Nesterov Accelerated Gradient** : 현재 방향으로 한번 가보고 그곳에서 gradient를 구한걸 가지고 accumulation 한다. 
    * <!-- $a_{t+1} \leftarrow \beta a_t + \nabla L(W_t - \eta \beta a_t)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a_%7Bt%2B1%7D%20%5Cleftarrow%20%5Cbeta%20a_t%20%2B%20%5Cnabla%20L(W_t%20-%20%5Ceta%20%5Cbeta%20a_t)">​
      * <!-- $\nabla L(W_t - \eta \beta a_t)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla%20L(W_t%20-%20%5Ceta%20%5Cbeta%20a_t)"> : Lookahead gradient
      * <!-- $a_{t+1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a_%7Bt%2B1%7D"> : accumulation
    * <!-- $W_{t+1} \leftarrow W_t - \eta a_{t+1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%5Cleftarrow%20W_t%20-%20%5Ceta%20a_%7Bt%2B1%7D">
  * **Adagrad** : (Adaptive) 파라미터가 지금까지 얼마나 변해왔는지 아닌지를 보고, 많이 변한 파라미터는 적게 변화시키고, 안 변한 파라미터는 많이 변화시키고 싶은 것
    * <!-- $W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%3D%20W_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_t%20%2B%20%5Cepsilon%7D%7D%20g_t">​​
      * <!-- $G_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_t"> : Sum of gradient squares, 지금까지 gradient가 얼마나 많이 변했는지를 제곱해서 더한 것, 학습 중에 계속 커지기 때문에 뒤로 갈수록 학습이 멈출 수도 있음
      * <!-- $\epsilon$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cepsilon"> : for numerical stability
  * **Adadelta** : Adagrad에서 learning rate이 <!-- $G_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_t">​ 의 역수로 표현됨으로써 생기는 monotonic한 decreasing property를 막는 방법, no learning rate, 사실 많이 사용되지는 않음
    * <!-- $G_t = \gamma G_{t-1} + (1 - \gamma) g_t^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_t%20%3D%20%5Cgamma%20G_%7Bt-1%7D%20%2B%20(1%20-%20%5Cgamma)%20g_t%5E2">​ : EMA(exponential moving average) of gradient squares
    * <!-- $W_{t+1} = W_t - \frac{\sqrt{H_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%3D%20W_t%20-%20%5Cfrac%7B%5Csqrt%7BH_%7Bt-1%7D%20%2B%20%5Cepsilon%7D%7D%7B%5Csqrt%7BG_t%20%2B%20%5Cepsilon%7D%7D%20g_t">
    * <!-- $H_t = \gamma H_{t-1} + (1 - \gamma) (\Delta W_t)^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t%20%3D%20%5Cgamma%20H_%7Bt-1%7D%20%2B%20(1%20-%20%5Cgamma)%20(%5CDelta%20W_t)%5E2">​​ : EMA of difference squares
  * **RMSprop** : Geoff Hinton의 강의에서 제안됨
    * <!-- $G_t = \gamma G_{t-1} + (1-\gamma) g_t^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=G_t%20%3D%20%5Cgamma%20G_%7Bt-1%7D%20%2B%20(1-%5Cgamma)%20g_t%5E2">​ : EMA of gradient squares
    * <!-- $W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%3D%20W_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_t%20%2B%20%5Cepsilon%7D%7D%20g_t">​
      * <!-- $\eta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ceta"> : stepsize
  * **Adam** : Adaptive Moment Estimation, 무난하게 사용하는 방법, adaptive learning rate approach와 Momentum 두 가지 방식을 결합한 것
    * <!-- $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m_t%20%3D%20%5Cbeta_1%20m_%7Bt-1%7D%20%2B%20(1-%5Cbeta_1)g_t">​ : Momentum
    * <!-- $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=v_t%20%3D%20%5Cbeta_2%20v_%7Bt-1%7D%20%2B%20(1%20-%20%5Cbeta_2)%20g_t%5E2"> : EMA of gradient squares
    * <!-- $W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} m_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_%7Bt%2B1%7D%20%3D%20W_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bv_t%20%2B%20%5Cepsilon%7D%7D%20%5Cfrac%7B%5Csqrt%7B1%20-%20%5Cbeta_2%5Et%7D%7D%7B1%20-%20%5Cbeta_1%5Et%7D%20m_t">​
      * <!-- $\eta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ceta"> : stepsize
      * <!-- $\epsilon$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cepsilon"> : 실제로 적용할 때, 입실론 값을 잘 조절하는 것이 중요하다.
  * [**RAdam**](https://github.com/LiyuanLucasLiu/RAdam)
  * [**AdamP**](https://github.com/clovaai/AdamP)

* Regularization
  * Early stopping : 오버피팅 전에 학습 종료하기
  * Parameter norm penalty : 부드러운(smoothness) 함수를 만들기 위함
  * Data augmentation : 주어진 데이터셋을 어떻게든 늘려서 사용하는 것
  * Noise robustness : 입력 데이터 또는 가중치에 노이즈를 넣는 것
  * Label Smoothing : mix-up, cutmix 등을 통해 decision boundary를 부드럽게 만드는 것
  * Dropout : 랜덤하게 가중치를 0으로 만드는 것, robust한 feature를 잡을 수 있기를 기대
  * Batch Normalization : 정규화하고자 하는 레이어의 statistics를 정규화하는 것

## 과제 수행 과정

* 필수과제2 - Optimization
  * https://pytorch.org/docs/stable/generated/torch.nn.Module.html
  * Model 클래스가 nn.Module을 상속하고 있기 때문에,  self.modules() 를 호출하면 nn.Module의 내장 메소드가 호출된다.
  * self.modules()를 호출하면 \__init__() 에서 선언된 nn.Linear 들을 전부 불러올 수 있는 것 같다.
  * (nn.Linear나 nn.Conv2d 등은 nn.Module을 상속하고 있다.)

## 도메인 특강

* 필요한 지식
  * 컴퓨터 공학의 기본 지식
  * ML 모델에 대한 기본기
  * 나머지는 필요하다는 것만 인지하면 자연스럽게 습득될 것이다.
* 공부 추천
  * Attention is all you need 여러 번 읽기
    * https://arxiv.org/abs/1706.03762
    * http://nlp.seas.harvard.edu/2018/04/03/attention.html
* NLP모델이 언어 디펜던시가 없어지는 방향으로 연구되고 있다.

## 피어 세션 정리

* 논문 리뷰 발표
  * (1) VGG : https://github.com/Barleysack/BoostCampPaperStudy/blob/main/VGG.pdf
  * (2) Batch Normalization : https://github.com/Barleysack/BoostCampPaperStudy/blob/main/BatchNormalization.pdf

## 마스터 클래스

* 읽어 볼 논문 추천 : [Visual Analytics in Deep Learning](https://fredhohman.com/visual-analytics-in-deep-learning/)

## 학습 회고

* 논문 리뷰 시간을 좀 더 알차게 쓰려면 어떻게 해야할까
* Attention is all you need 가 여러 번 읽을만큼 중요하다는 것을 깨달았다. 꼭 다시 (제대로) 읽어봐야겠다.
