# 부스트캠프 3일차

- [부스트캠프 3일차](#부스트캠프-3일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 복습 내용](#강의-복습-내용)
    - [[7강] 통계학](#7강-통계학)
    - [[8강] 베이즈 통계학](#8강-베이즈-통계학)
    - [[9강] CNN](#9강-cnn)
  - [피어 세션 정리](#피어-세션-정리)
  - [피어세션이 피어씁니다.](#피어세션이-피어씁니다)
  - [과제 수행 과정](#과제-수행-과정)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/4 (수)
  - [x] Python: 선택과제 1
  - [x] AI Math: 필수퀴즈 7,8,9강
  - [x] 피어세션이 피어씁니다 (오후6시~7시)
* 오늘까지 모더레이터 담당!

## 강의 복습 내용

### [7강] 통계학

* 통계적 모델링 : 적절한 가정 위에서 근사적으로 확률분포 추정
  * **모수적(parametric) 방법론** : 데이터가 특정 확률 분포를 따른다고 선험적으로 가정한 후 그 분포를 결정하는 모수(parameter)를 추정하는 방법
    * ex) 정규분포의 모수 : 평균, 분산
  * **비모수(nonparametric) 방법론** : 특정 확률분포를 가정하지 않고 **데이터에 따라** 모델의 구조 및 모수의 개수가 유연하게 **바뀌는 것**
    * 기계학습의 많은 방법이 비모수 방법론

* **최대가능도 추정법(maximum likelihoof estimation, MLE)** : 이론적으로 가장 가능성이 높은 모수를 추정하는 방법
  * 가능도(likelihood) 함수 : 모수 <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctheta">를 따르는 분포가 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">를 관찰할 가능성(확률X)
    * <!-- $\hat{\theta}_{MLE} = \underset{\theta}{argmax} L(\theta ; \bold{x}) = \underset{\theta}{argmax} P(\bold{x} | \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Ctheta%7D_%7BMLE%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7Bargmax%7D%20L(%5Ctheta%20%3B%20%5Cbold%7Bx%7D)%20%3D%20%5Cunderset%7B%5Ctheta%7D%7Bargmax%7D%20P(%5Cbold%7Bx%7D%20%7C%20%5Ctheta)">​
  * 데이터 집합 <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X">​가 독립적으로 추출되었을 경우 **로그가능도**를 최적화
    * <!-- $L(\theta; X) = \Pi_{i=1}^{n} P(x_i | \theta) \Rightarrow \log L(\theta; X) = \sum_{i=1}^{n} \log P(x_i | \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L(%5Ctheta%3B%20X)%20%3D%20%5CPi_%7Bi%3D1%7D%5E%7Bn%7D%20P(x_i%20%7C%20%5Ctheta)%20%5CRightarrow%20%5Clog%20L(%5Ctheta%3B%20X)%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Clog%20P(x_i%20%7C%20%5Ctheta)">
    * 로그가능도를 이용하면 미분 연산의 연산량을 <!-- $O(n^2)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O(n%5E2)">에서 <!-- $O(n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O(n)">으로 줄일 수 있다.
    * 경사하강법의 경우 음의 로그가능도(negative log-likelihood)를 최적화
  * 딥러닝에서 최대가능도 추정법 예시
    * 분류문제에서 소프트맥스 벡터는 카테고리분포의 모수 (<!-- $p_1, ..., p_K$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p_1%2C%20...%2C%20p_K">)를 모델링
    * 원핫벡터로 표현한 정답레이블 <!-- $y = (y_1, ..., y_K)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20(y_1%2C%20...%2C%20y_K)">​ 을 관찰데이터로 이용해 확률분포인 소프트맥스 벡터의 로그가능도를 최적화
    * <!-- $\hat{\theta}_{MLE} = \underset{\theta}{argmax} \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log (MLP_\theta (\bold{x}_i)_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Ctheta%7D_%7BMLE%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7Bargmax%7D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20y_%7Bi%2Ck%7D%20%5Clog%20(MLP_%5Ctheta%20(%5Cbold%7Bx%7D_i)_k)">
  * MLE로 추정하는 모델학습방법론은 확률분포의 거리를 최적화하는 것과 밀접한 관련이 있다.
    * **쿨백-라이블러 발산(Kullback-Leibler Divergence, KL)** : 두 확률분포가 어느 정도 닮았는지를 나타내는 척도
      * 이산확률변수일 떄 : <!-- $KL(P||Q) = \sum_{x \in X} P(x) \log (\frac{P(x)}{Q(x)})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=KL(P%7C%7CQ)%20%3D%20%5Csum_%7Bx%20%5Cin%20X%7D%20P(x)%20%5Clog%20(%5Cfrac%7BP(x)%7D%7BQ(x)%7D)">
      * 연속확률변수일 때 : <!-- $KL(P||Q) = \int_{X} P(x) \log (\frac{P(x)}{Q(x)}) dx$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=KL(P%7C%7CQ)%20%3D%20%5Cint_%7BX%7D%20P(x)%20%5Clog%20(%5Cfrac%7BP(x)%7D%7BQ(x)%7D)%20dx">
      * 분해 : <!-- $KL(P||Q) = - E_{x \sim P(x)} [\log Q(x)] + E_{x \sim P(x)} [\log P(x)]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=KL(P%7C%7CQ)%20%3D%20-%20E_%7Bx%20%5Csim%20P(x)%7D%20%5B%5Clog%20Q(x)%5D%20%2B%20E_%7Bx%20%5Csim%20P(x)%7D%20%5B%5Clog%20P(x)%5D">
    * 분류문제에서 정답레이블을 P, 모델 예측을 Q라 두면 최대가능도 추정법은 쿨백-라이블러 발산을 최소화하는 것과 같다.

### [8강] 베이즈 통계학

* **조건부 확률** <!-- $P(A|B) = \frac{P(A \cap B)}{P(B)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(A%7CB)%20%3D%20%5Cfrac%7BP(A%20%5Ccap%20B)%7D%7BP(B)%7D">​ : 사건 B가 일어난 상황에서 사건 A​​가 발생할 확률
  * <!-- $P(B|A) = \frac{P(A \cap B)}{P(A)} = P(B) \frac{P(A|B)}{P(A)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(B%7CA)%20%3D%20%5Cfrac%7BP(A%20%5Ccap%20B)%7D%7BP(A)%7D%20%3D%20P(B)%20%5Cfrac%7BP(A%7CB)%7D%7BP(A)%7D">

* **베이즈 정리** : 데이터가 새로 추가될 때 조건부 확률을 이용하여 정보를 갱신하는 방법
  * <!-- $P(\theta | D) = P(\theta) \frac{P(D | \theta)}{P(D)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Ctheta%20%7C%20D)%20%3D%20P(%5Ctheta)%20%5Cfrac%7BP(D%20%7C%20%5Ctheta)%7D%7BP(D)%7D">
    * <!-- $P(\theta | D)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Ctheta%20%7C%20D)">​​ : 사후확률(posterior), 데이터를 관찰했을 때 이 모수나 가설(hypothesis)이 성립할 확률
    * <!-- $P(\theta), P(\neg \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Ctheta)%2C%20P(%5Cneg%20%5Ctheta)"> : 사전확률(prior), 모델링하기 이전에 모수나 가설에 대해 주어진 확률
    * <!-- $P(D | \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D%20%7C%20%5Ctheta)">​​ : 가능도(likelihood) : 현재 주어진 모수, 가정에서 이 데이터가 관찰될 확률
    * <!-- $P(D)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D)"> : Evidence, 데이터 자체의 분포
  * 가능도와 Evidence를 통해 사전확률을 사후확률로 업데이트 한다.
  * 새로운 데이터가 들어왔을 때, 앞서 계산한 사후확률을 사전확률로 사용하여 사후확률을 갱신할 수 있다.

* 용어
  * True Positive = <!-- $P(D | \theta) P(\theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D%20%7C%20%5Ctheta)%20P(%5Ctheta)">
  * False Positive (1종 오류) = <!-- $P(D | \neg \theta) P(\neg \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D%20%7C%20%5Cneg%20%5Ctheta)%20P(%5Cneg%20%5Ctheta)">
  * False Negative (2종 오류) = <!-- $P(\neg D | \theta) P(\theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Cneg%20D%20%7C%20%5Ctheta)%20P(%5Ctheta)">
  * True Negative = <!-- $P(\neg D | \neg \theta) P(\neg \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Cneg%20D%20%7C%20%5Cneg%20%5Ctheta)%20P(%5Cneg%20%5Ctheta)">
  * 정밀도(Precision) = <!-- $P(\theta | D)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Ctheta%20%7C%20D)">​​ = TP / (TP + FP)
  * 민감도(Recall) = <!-- $P(D | \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D%20%7C%20%5Ctheta)">
  * 오탐(False alarm) = <!-- $P(D | \neg \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(D%20%7C%20%5Cneg%20%5Ctheta)">
  * 특이도(specificity) = <!-- $P(\neg D | \neg \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(%5Cneg%20D%20%7C%20%5Cneg%20%5Ctheta)">​

* 인과관계
  * 조건부 확률만 가지고 인과관계(causality)를 추론하는 것은 불가능
  * 조정(intervention) 효과를 통해 중간 개입을 제거할 수 있다.

### [9강] CNN

* 커널(kernel)을 입력벡터 상에서 움직이며 선형모델과 합성함수가 적용되는 구조

* Convolution 연산의 수학적인 의미
  * continuous : <!-- $[f * g](x) = \int_{R^d} f(z)g(x+z)dz = \int_{R^d} f(x+z)g(z)dz = [g*f](x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Bf%20*%20g%5D(x)%20%3D%20%5Cint_%7BR%5Ed%7D%20f(z)g(x%2Bz)dz%20%3D%20%5Cint_%7BR%5Ed%7D%20f(x%2Bz)g(z)dz%20%3D%20%5Bg*f%5D(x)">​
  * discrete : <!-- $[f * g](i) = \sum_{a \in Z^d} f(a)g(i+a) = \sum_{a \in Z^d} f(i + a)g(a) = [g*f](i)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Bf%20*%20g%5D(i)%20%3D%20%5Csum_%7Ba%20%5Cin%20Z%5Ed%7D%20f(a)g(i%2Ba)%20%3D%20%5Csum_%7Ba%20%5Cin%20Z%5Ed%7D%20f(i%20%2B%20a)g(a)%20%3D%20%5Bg*f%5D(i)">
  * 커널을 이용해 신호(signal)를 국소적(local)으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것

* 다양한 차원에서 계산 가능 (f는 커널, g는 입력)
  * 1D-conv : <!-- $[f * g](i) = \sum_{p=1}^{d} f(p)g(i+p)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Bf%20*%20g%5D(i)%20%3D%20%5Csum_%7Bp%3D1%7D%5E%7Bd%7D%20f(p)g(i%2Bp)">
  * 2D-conv : <!-- $[f * g](i, j) = \sum_{p, q} f(p, q)g(i+p, j+q)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Bf%20*%20g%5D(i%2C%20j)%20%3D%20%5Csum_%7Bp%2C%20q%7D%20f(p%2C%20q)g(i%2Bp%2C%20j%2Bq)">
  * 3D-conv : <!-- $[f * g](i, j, k) = \sum_{p, q, r} f(p, q, r)g(i+p, j+q, k+r)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Bf%20*%20g%5D(i%2C%20j%2C%20k)%20%3D%20%5Csum_%7Bp%2C%20q%2C%20r%7D%20f(p%2C%20q%2C%20r)g(i%2Bp%2C%20j%2Bq%2C%20k%2Br)">

* Convolution 연산의 역전파 : 역전파 계산시에도 convolution 연산
  * <!-- $\frac{\partial}{\partial x} [f*g](x) = \frac{\partial}{\partial x} \int_{R^d} f(y) g(x-y)dy = \int_{R^d} f(y) \frac{\partial g}{\partial x} (x-y)dy = [f*g'](x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%20%5Bf*g%5D(x)%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%20%5Cint_%7BR%5Ed%7D%20f(y)%20g(x-y)dy%20%3D%20%5Cint_%7BR%5Ed%7D%20f(y)%20%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20x%7D%20(x-y)dy%20%3D%20%5Bf*g'%5D(x)">

## 피어 세션 정리

* Q. 활성함수와 비선형함수의 차이가 뭔가요?
  * 활성함수를 써서 비선형 형태로 만드는 것이다.실수값을 입력으로 받아 다시 실수값을 반환해주는 비선형함수

* Q. 소프트맥스 함수는 활성함수가 아닌건가요?
  * 소프트맥스 함수도 활성함수다.

* Q. 분류 문제에서 softmax를 계산하는 게 조건부 확률을 계산하는 것과 같다는게 무슨 의미인가요?
  * 입력데이터가 주어졌을 때, 그 데이터에 따라 softmax로 나온 결과가 각 카테고리별 확률값이니까 결국에는 이 데이터에 대한 카테고리가 나올 조건부확률이다.
  * P(Y|X) → 데이터 X가 카테고리 Y일 확률 → softmax와 상동

* Q. 몬테카를로 샘플링이 뭔가요?
  * 특징 : 가능한 모든 수를 시도하는 것이 전제로 들어감, 약간 확률적인 완전탐색/브루트포스 같은 느낌?
  * 독립추출이 보장되면, 대수의 법칙(law of large number)에 의해 수렴성 보장 = 수를 많이 뽑으면 정답에 가까워질 것이라는 뜻

* Q. 9강 CNN 수식이 무슨 뜻인가요?
  * '*' 는 convolution 연산을 뜻하고, kernel을 움직이면서 더한다는 것을 수학적으로 표현한 것이다.
  * '+' / '-' 의 뜻 : 함수의 좌우 반전!

* CNN Backpropagation : <!-- $\frac{\partial}{\partial x} [f*g](x) = \frac{\partial}{\partial x} \int_{R^d} f(y) g(x-y)dy = \int_{R^d} f(y) \frac{\partial g}{\partial x} (x-y)dy = [f*g'](x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%20%5Bf*g%5D(x)%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%20%5Cint_%7BR%5Ed%7D%20f(y)%20g(x-y)dy%20%3D%20%5Cint_%7BR%5Ed%7D%20f(y)%20%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20x%7D%20(x-y)dy%20%3D%20%5Bf*g'%5D(x)">
  * forward pass : <!-- $O_i = \sum_{j} w_j x_{i+j-1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O_i%20%3D%20%5Csum_%7Bj%7D%20w_j%20x_%7Bi%2Bj-1%7D">
  * backward pass : <!-- $\frac{\partial L}{\partial W_i} = \sum_{j} \delta_j x_{i+j-1}, \frac{\partial L}{\partial x_i} = \sum_{j} \delta_j w_{i-j+1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W_i%7D%20%3D%20%5Csum_%7Bj%7D%20%5Cdelta_j%20x_%7Bi%2Bj-1%7D%2C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20x_i%7D%20%3D%20%5Csum_%7Bj%7D%20%5Cdelta_j%20w_%7Bi-j%2B1%7D">​​
  * ![image](https://user-images.githubusercontent.com/35680202/128604387-2be5b830-65d3-4c86-bb14-07a2ff97477c.png)

## 피어세션이 피어씁니다.

* 다른 조 좋은 피어 규칙
  * 모더레이터가 아침에 슬랙에 to-do list 올리기
  * 학습 정리 노트 공유
  * 다음 주부터 스터디
  * 파트를 나눠서 발표

## 과제 수행 과정

* 선택과제1 : sympy를 처음 써봐서 당황했는데 사용법은 생각보다 간단했다. 행렬곱할 때, np.dot() 쓰는걸 잊지 말자. 

## 학습 회고

* 주말에 추가자료로 올려진 통계학 강의를 들어야겠다.
* 완벽히 이해하지 못한 것들 꼭 질문해야겠다. 혼자 고민하는 것보다 진짜 훨씬 낫다.

