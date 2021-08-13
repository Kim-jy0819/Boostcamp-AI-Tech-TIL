# 부스트캠프 10일차

## 오늘 일정 정리

* 8/13 (금)
  - [x] DL Basic
    - [x] (9강) Generative models 1
    - [x] (10강) Generative models 2
  - [x] 스페셜 피어세션 (16:00~17:00) : 한 주간의 학습에 대해서, 각 조의 피어세션에 대해서 이야기
  - [x] 피어세션 (17:00~18:00) : 팀 회고
  - [x] 마스터 클래스 : 18:00~19:00 최성준 마스터님 - ‘Learning what we do not know in Deep Learning’

## 강의 내용 정리

### [9강] Generative Models 1

* Generative model : 생성 모델, probability distribution <!-- $p(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x)"> 를 배우는 것이다.
  * 기능
    * Generation : <!-- $x_{new} \sim p(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bnew%7D%20%5Csim%20p(x)">​​ 을 sampling하면, 마치 강아지같은 이미지를 얻을 수 있다.
    * Density estimation : <!-- $p(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x)"> 로 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">​가 강아지와 비슷한지 아닌지를 구분할 수 있다. (anomaly detection에 사용될 수 있다.)
    * Unsupervised representation learning : 강아지 이미지에는 보통 귀, 꼬리가 있다는 특성을 배우는 것 (feature learning)
  * 종류
    * explicit (generative) model : 입력이 주어졌을 때, 이 입력에 대한 확률값을 얻어낼 수 있는 모델
    * implicit (generative) model : 단순히 generation만 하는 모델
  * 핵심 : <!-- $p(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x)">​를 어떻게 만들(표현할) 것인가?

* Basic Discrete Distributions : 관심있어하는 값들이 finite set인 경우
  * Bernoulli distribution : (biased) coin flip, 0 또는 1(head or tail)이 나옴
    * <!-- $D = \{Heads, Tails \}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D%20%3D%20%5C%7BHeads%2C%20Tails%20%5C%7D">
    * <!-- $P(X = Heads) = p$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(X%20%3D%20Heads)%20%3D%20p"> 라고 하면, <!-- $P(X = Tails) = 1 - p$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(X%20%3D%20Tails)%20%3D%201%20-%20p"> 가 된다.
    * <!-- $X \sim Ber(p)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X%20%5Csim%20Ber(p)">​ 라고 표기한다.
    * 확률을 표현하는데 한 개의 파라미터가 필요하다.
  * Categorical distribution : (biased) m-sided dice
    * <!-- $D = \{ 1, ..., m \}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D%20%3D%20%5C%7B%201%2C%20...%2C%20m%20%5C%7D">
    * <!-- $P(Y = i) = p_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(Y%20%3D%20i)%20%3D%20p_i"> 라고 하면, <!-- $\sum_{i=1}^{m} p_i = 1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20p_i%20%3D%201"> 이다.
    * <!-- $Y \sim Cat(p_1, ..., p_m)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y%20%5Csim%20Cat(p_1%2C%20...%2C%20p_m)">​ 라고 표기한다.
    * 확률을 표현하는데 m-1개의 파라미터가 필요하다.

* 예제
  1. (한 픽셀에 대한) RGB joint distribution 을 만들어보자
     * <!-- $(r, g, b) \sim p(R, G, B)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=(r%2C%20g%2C%20b)%20%5Csim%20p(R%2C%20G%2C%20B)">
     * 경우의 수 : 256 x 256 x 256
     * 확률을 표현하는데 필요한 파라미터 수 : (256 x 256 x 256) - 1
     * 즉, 하나의 RGB픽셀을 fully discribe하기 위해서 필요한 파라미터의 숫자가 엄청 크다.
  2. n개의 binary pixels (<!-- $X_1, ..., X_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_1%2C%20...%2C%20X_n">)를 가지는 binary image 하나가 있다고 하자
     * 경우의 수 : <!-- $2^n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=2%5En">
     * 확률 <!-- $p(x_1,...,x_n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C...%2Cx_n)">을 표현하는데 필요한 파라미터 수 : <!-- $2^n - 1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=2%5En%20-%201">​

* Structure Through Independence
  * 동기 : 기계학습에서 파라미터 수가 늘어나면 학습은 더 어렵다.
  * 만약 위 예제 2번에서 <!-- $X_1, ..., X_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_1%2C%20...%2C%20X_n">​개의 픽셀들이 모두 independent 하다고 생각하면 어떨까? (말이 안 되는 가정이긴 함)
    * <!-- $p(x_1, ..., x_n) = p(x_1) p(x_2) ... p(x_n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C%20...%2C%20x_n)%20%3D%20p(x_1)%20p(x_2)%20...%20p(x_n)"> 으로 나타낼 수 있다.
    * 가능한 경우의 수 : <!-- $2^n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=2%5En"> (위와 똑같다.)
    * 확률 <!-- $p(x_1,...,x_n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C...%2Cx_n)">을 표현하는데 필요한 파라미터 수 : <!-- $n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n">

* Conditional Independence
  * 동기
    * 픽셀들이 fully dependent 하면 너무 많은 파라미터가 필요하다.
    * 픽셀들이 모두 independent 하면 파라미터는 줄어들어서 좋은데, 표현할 수 있는 이미지가 너무 적다. (일반적으로 우리가 아는 이미지를 전혀 만들 수 없다.)
    * 이 중간 어딘가 적절한 것을 찾고 싶다.
  * 핵심
    * Chain rule : n개의 joint distribution을 n개의 conditional distribution으로 표현해주는 것 (independent한 것과 관련 없이 항상 만족한다.)
      * <!-- $p(x_1, ..., x_n) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) ... p(x_n|x_1, ..., x_{n-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C%20...%2C%20x_n)%20%3D%20p(x_1)%20p(x_2%7Cx_1)%20p(x_3%7Cx_1%2C%20x_2)%20...%20p(x_n%7Cx_1%2C%20...%2C%20x_%7Bn-1%7D)">
    *  Bayes' rule : 
      * <!-- $p(x|y) = \frac{p(x, y)}{p(y)} = \frac{p(y|x) p(x)}{p(y)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x%7Cy)%20%3D%20%5Cfrac%7Bp(x%2C%20y)%7D%7Bp(y)%7D%20%3D%20%5Cfrac%7Bp(y%7Cx)%20p(x)%7D%7Bp(y)%7D">
    * Conditional independence 
      * 만약 <!-- $x \perp y | z$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%5Cperp%20y%20%7C%20z"> 이면, <!-- $p(x|y,z) = p(x|z)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x%7Cy%2Cz)%20%3D%20p(x%7Cz)"> 이다.
      * z가 주어졌을 때, x와 y가 independent하다면(conditional independent), x를 표현하는데에 z가 주어지면 y는 상관이 없어진다.(뒷단의 conditional 부분을 날려줄 수 있다.)
  * 목표 : Conditional independence와 Chain rule을 잘 섞어서 좋은 모델을 만들자
  * 방법
    * Chain rule 을 사용하여 표현하자
      * <!-- $p(x_1, ..., x_n) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) ... p(x_n|x_1, ..., x_{n-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C%20...%2C%20x_n)%20%3D%20p(x_1)%20p(x_2%7Cx_1)%20p(x_3%7Cx_1%2C%20x_2)%20...%20p(x_n%7Cx_1%2C%20...%2C%20x_%7Bn-1%7D)">
      * 확률을 표현하는데 필요한 파라미터 수?
        * <!-- $p(x_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1)"> : 1개
        * <!-- $p(x_2 | x_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%20%7C%20x_1)"> : 2개 => <!-- $p(x_2|x_1 = 0)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%7Cx_1%20%3D%200)">, <!-- $p(x_2 | x_1 = 1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%20%7C%20x_1%20%3D%201)">
        * <!-- $p(x_3 | x_1, x_2)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%20%7C%20x_1%2C%20x_2)"> : 4개 => <!-- $p(x_3|x_1 = 0, x_2=0)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%7Cx_1%20%3D%200%2C%20x_2%3D0)">, <!-- $p(x_3|x_1 = 1, x_2=0)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%7Cx_1%20%3D%201%2C%20x_2%3D0)">, <!-- $p(x_3|x_1 = 0, x_2=1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%7Cx_1%20%3D%200%2C%20x_2%3D1)">, <!-- $p(x_3|x_1 = 1, x_2=1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%7Cx_1%20%3D%201%2C%20x_2%3D1)">
        * 즉, <!-- $2^n - 1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=2%5En%20-%201">개 (fully independent와 똑같다)
    * Markov assumption 을 가정하자
      * i+1 번째 픽셀은 i 번째 픽셀에만 dependent 하다.
      * <!-- $X_{i+1} \perp X_1,...,X_{i-1} | X_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_%7Bi%2B1%7D%20%5Cperp%20X_1%2C...%2CX_%7Bi-1%7D%20%7C%20X_i">
    * Conditional independence 에 의해​,
      * <!-- $p(x_1, ..., x_n) = p(x_1) p(x_2|x_1) p(x_3|x_2) ... p(x_n|x_{n-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C%20...%2C%20x_n)%20%3D%20p(x_1)%20p(x_2%7Cx_1)%20p(x_3%7Cx_2)%20...%20p(x_n%7Cx_%7Bn-1%7D)">​​​ 으로 바뀐다.
      * 확률을 표현하는데 필요한 파라미터 수?
        * <!-- $p(x_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1)"> : 1개
        * <!-- $p(x_2 | x_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%20%7C%20x_1)"> : 2개 => <!-- $p(x_2|x_1 = 0)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%7Cx_1%20%3D%200)">, <!-- $p(x_2 | x_1 = 1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_2%20%7C%20x_1%20%3D%201)">
        * <!-- $p(x_3 | x_2)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%20%7C%20x_2)"> : 2개 => <!-- $p(x_3|x_2 = 0)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%7Cx_2%20%3D%200)">, <!-- $p(x_3 | x_2 = 1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_3%20%7C%20x_2%20%3D%201)">
        * 즉, <!-- $2n - 1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=2n%20-%201">​개

* Auto-regressive Model
  * 핵심
    * 하나의 정보가 이전 정보에 dependent 한 특징을 가지는 모델들을 전반적으로 지칭
    * 위의 Conditional Independence를 잘 이용한 모델도 포함
    * 이전 정보 n개에 dependent한 모델을 ar-n 모델이라고 함
  * Neural Autoregressive Density Estimator(NADE)
    * 방법
      * i 번째 픽셀이 1부터 i-1 번째 픽셀에 dependent 하다고 가정
      * <!-- $p(x_i | x_{1 : i-1}) = \sigma (\alpha_{i} h_i + b_i)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_i%20%7C%20x_%7B1%20%3A%20i-1%7D)%20%3D%20%5Csigma%20(%5Calpha_%7Bi%7D%20h_i%20%2B%20b_i)">​ (이때, <!-- $h_i = \sigma (W_{< i} x_{1 : i-1} + c)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_i%20%3D%20%5Csigma%20(W_%7B%3C%20i%7D%20x_%7B1%20%3A%20i-1%7D%20%2B%20c)"> ​)
      * 100번째 픽셀에 대한 확률분포를 만들기 위해서 99개의 이전 입력들을 받을 수 있는 neural network가 필요하다.
    * 특징
      * explicit model 이다. (입력이 들어오면 이 입력의 확률을 구할 수 있다.)
    * 출력
      * binary output이면 그냥 sigmoid 통과한다.
      * continuous output이면 마지막 레이어에 가우시안 mixture 모델을 사용해서 continuous한 distribution을 만든다.
  * Pixel RNN
    * 방법
      * <!-- $n \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Ctimes%20n">​ RGB 이미지가 있을 때,
      * <!-- $p(x) = \Pi_{i=1}^{n^2} p(x_{i,R} | x_{< i}) p(x_{i,B} | x_{< i}, x_{i, R}) p(x_{i,B} | x_{< i}, X_{i,R}, X_{i, G})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x)%20%3D%20%5CPi_%7Bi%3D1%7D%5E%7Bn%5E2%7D%20p(x_%7Bi%2CR%7D%20%7C%20x_%7B%3C%20i%7D)%20p(x_%7Bi%2CB%7D%20%7C%20x_%7B%3C%20i%7D%2C%20x_%7Bi%2C%20R%7D)%20p(x_%7Bi%2CB%7D%20%7C%20x_%7B%3C%20i%7D%2C%20X_%7Bi%2CR%7D%2C%20X_%7Bi%2C%20G%7D)">
        * <!-- $p(x_{i,R} | x_{< i})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_%7Bi%2CR%7D%20%7C%20x_%7B%3C%20i%7D)"> : i 번째 픽셀의 R 에 대한 확률​
        * <!-- $p(x_{i,B} | x_{< i}, x_{i, R})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_%7Bi%2CB%7D%20%7C%20x_%7B%3C%20i%7D%2C%20x_%7Bi%2C%20R%7D)"> : i 번째 픽셀의 G 에 대한 확률
        * <!-- $p(x_{i,B} | x_{< i}, X_{i,R}, X_{i, G})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_%7Bi%2CB%7D%20%7C%20x_%7B%3C%20i%7D%2C%20X_%7Bi%2CR%7D%2C%20X_%7Bi%2C%20G%7D)"> : i 번째 픽셀의 B 에 대한 확률
    * 특징
      * auto-regressive model을 fully connected layer로 만든 것이 아니라 RNN을 이용한다.
      * ordering 방법에 따라
        * ![image](https://user-images.githubusercontent.com/35680202/129298636-069ae7a5-c3df-4d94-9457-aecd639745a5.png)
        * Row LSTM : 위쪽에 있는 정보 활용
        * Diagonal BiLSTM : 이전 정보들을 다 활용

### [10강] Generative Models 2

* **Variational Auto-encoder(VAE)**
  * Variational inference (VI) : posterior distribution 을 제일 잘 근사할 수 있는 variational distribution 을 찾는 일련의 과정
    * Posterior distribution <!-- $p_{\theta} (z|x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p_%7B%5Ctheta%7D%20(z%7Cx)">​
      * observation 이 주어졌을 때, 내가 관심있어하는 random variable의 확률분포 (이 반대를 likelihood 라고 부른다.) 
      * <!-- $z$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=z">​​는 latent vector(잠재벡터)
    * Variational distribution <!-- $q_{\phi} (z|x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=q_%7B%5Cphi%7D%20(z%7Cx)">​​
      * 일반적으로 posterior distribution을 계산하기 불가능할 때가 많다.
      * 학습&최적화를 통해 posterior distribution 를 근사하는 분포가 variational distribution
    * 방법
      * 마치 target을 모르는데 loss function을 찾고자 하는 것
      * KL divergence 를 최소화하는 variational distribution을 찾는다.
      * <!-- $\ln p_{\theta}(D)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cln%20p_%7B%5Ctheta%7D(D)">
        * <!-- $= E_{q_{\phi}(z|x)} [\ln p_{\theta}(x)]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20p_%7B%5Ctheta%7D(x)%5D">​​
        * <!-- $= E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z)}{p_{\theta}(z|x)}]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%7D%7Bp_%7B%5Ctheta%7D(z%7Cx)%7D%5D">​
        * <!-- $= E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z) q_{\phi}(z|x)}{q_{\phi}(z|x) p_{\theta}(z|x)}]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%20q_%7B%5Cphi%7D(z%7Cx)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%20p_%7B%5Ctheta%7D(z%7Cx)%7D%5D">​​
        * <!-- $= E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)}] + E_{q_{\phi}(z|x)} [\ln \frac{q_{\phi}(z|x)}{p_{\theta}(z|x)}]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%5D%20%2B%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%7Bp_%7B%5Ctheta%7D(z%7Cx)%7D%5D">​​​​
        * <!-- $= E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)}] + D_{KL}(q_{\phi}(z|x) || p_{\theta}(z|x))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%5D%20%2B%20D_%7BKL%7D(q_%7B%5Cphi%7D(z%7Cx)%20%7C%7C%20p_%7B%5Ctheta%7D(z%7Cx))">
          * ELBO(↑) : <!-- $E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)}]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%5D">​
          * Objective(↓) : <!-- $D_{KL}(q_{\phi}(z|x) || p_{\theta}(z|x))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D_%7BKL%7D(q_%7B%5Cphi%7D(z%7Cx)%20%7C%7C%20p_%7B%5Ctheta%7D(z%7Cx))">​​​
      * VI는 ELBO를 최대화시킴으로써 (intractable한) Objective를 최소화시킨다.
    * ELBO(Evidence of Lower BOund)
      * <!-- $E_{q_{\phi}(z|x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)}]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5B%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%2C%20z)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%5D">​​​
        * <!-- $= \int \ln \frac{p_{\theta}(x|z) p(z)}{q_{\phi}(z|x)} q_{\phi}(z|x) dz$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20%5Cint%20%5Cln%20%5Cfrac%7Bp_%7B%5Ctheta%7D(x%7Cz)%20p(z)%7D%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20q_%7B%5Cphi%7D(z%7Cx)%20dz">
        * <!-- $= E_{q_{\phi}(z|x)} [p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x) || p(z))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3D%20E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5Bp_%7B%5Ctheta%7D(x%7Cz)%5D%20-%20D_%7BKL%7D(q_%7B%5Cphi%7D(z%7Cx)%20%7C%7C%20p(z))">
          * Reconstruction Term : <!-- $E_{q_{\phi}(z|x)} [p_{\theta}(x|z)]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E_%7Bq_%7B%5Cphi%7D(z%7Cx)%7D%20%5Bp_%7B%5Ctheta%7D(x%7Cz)%5D"> => 인코더를 통해서 x를 latent space로 보냈다가 다시 디코더로 돌아오는 auto-encoder의 reconstruction loss를 줄이는 것이 Reconstruction Term
          * Prior Fitting Term : <!-- $D_{KL}(q_{\phi}(z|x) || p(z))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D_%7BKL%7D(q_%7B%5Cphi%7D(z%7Cx)%20%7C%7C%20p(z))"> => x 들을 latent space로 올려놓았을 때 점들이 이루는 분포가 내가 가정하는 사전분포(prior distribution)와 비슷하게 만들어주는 Term
    * 한계
      * intractable model 이다. (implicit model)
      * 미분 가능한 prior fitting term 을 사용해야 하므로, 다양한 latent prior distribution을 사용할 수 없다. (그래서 대부분의 경우 isotropic Gaussian 을 사용한다.)

* **Adversarial Auto-encoder(AAE)**
  * 방법
    * GAN을 사용해서 latent distribution 사이의 분포를 맞춰주는 것
    * Variational Autoencoder의 prior fitting term을 GAN의 objective로 바꿔버린 것

* **Generative Adversarial Network(GAN)**
  * ![image](https://user-images.githubusercontent.com/35680202/129310604-d6cc097f-5461-47ea-a226-6efba53e6a59.png)
  * A two player minimax game between generator and discriminator
    * <!-- $\underset{G}{\min} \underset{D}{\max} V(D, G) = E_{x \sim p_{data}(x)} [\log D(x)] + E_{z \sim p_z(z)} [\log(1 - D(G(z)))]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cunderset%7BG%7D%7B%5Cmin%7D%20%5Cunderset%7BD%7D%7B%5Cmax%7D%20V(D%2C%20G)%20%3D%20E_%7Bx%20%5Csim%20p_%7Bdata%7D(x)%7D%20%5B%5Clog%20D(x)%5D%20%2B%20E_%7Bz%20%5Csim%20p_z(z)%7D%20%5B%5Clog(1%20-%20D(G(z)))%5D">​
  * GAN Objective
    * For discriminator :
      * ![image](https://user-images.githubusercontent.com/35680202/129356148-dd5facb3-b7cc-4fd5-a755-0f5e146649fb.png)
    * For generator :
      * ![image](https://user-images.githubusercontent.com/35680202/129356264-918326bb-3f1e-4bfc-ad9a-ef9370ad641b.png)
  * implicit model

* DCGAN
  * GAN 이 MLP를 이용했다면 DCGAN 에서는 이미지 도메인으로 했다.
  * Deconvolution layer로 generator를 만들었다.
  * 여러 좋은 테크닉 사용 : leaky ReLU, 적절한 하이퍼파라미터 등

* Info-GAN
  * ![image](https://user-images.githubusercontent.com/35680202/129311554-f7c1f65c-7ec6-4244-8dbd-e5cdafdca39c.png)
  * class c 라는 auxiliary class를 랜덤하게 집어넣는다. (랜덤한 one-hot 벡터)
  * 마치 multi-modal distribution을 학습하는 것을 c라는 벡터를 통해서 잡아주는 역할

* Text2Image
  * 문장이 주어지면 이미지를 만드는 것

* Puzzle-GAN
  * 이미지 안의 subpatch 들이 있으면, 원래 이미지로 복원하는 것

* CycleGAN
  * GAN 구조를 사용하지만 이미지 사이의 도메인을 바꿀 수 있는 것
  * Cycle-consistency loss : 꼭 알아두기!
    * ![image](https://user-images.githubusercontent.com/35680202/129312171-87e97416-f4ee-4d6a-89f3-c257155ffe39.png)
    * GAN 구조가 2개가 들어감

* Star-GAN
  * 이미지를 단순히 다른 도메인으로 바꾸는 것이 아니라 내가 control 할 수 있게 하는 것

* Progressive-GAN
  * 고차원의 이미지를 잘 만들 수 있는 방법론
  * 4x4 부터 시작해서 1024x1024 까지 고해상도 이미지로 점점 늘려나가면서 학습하는 것

## 스페셜 피어 세션

* 피어세션 내용 공유
  * 근황토크
  * 노션에 사전질문올리고, 시작할 때 강의나 과제 질문 & 답변
  * 서로 필요한 것들, 찾아본 것들 공유하면 시간이 끝난다.
  * 당일 모더레이터분들이 코드 리뷰, 다르게 푼 분이 계시면 공유
  * 시간이 남으면 강의에서 헷갈렸던 부분 같이 고민, 멘토링 때 여쭤보고 싶은걸 논의해본다.
  * 스터디는 말은 나왔는데 아직은 정해진게 없는 상황이다.
  * 선택과제도 하신 분들거 보면서 짧게 짧게 했다.

* 멘토링 내용 공유
  * 현업에 관련된 얘기 : 직무, 구체적인 방향 등
  * 미리 적어놓으면 강의내용도 질의응답도 해주신다.
  * 진로나 대학원 궁금증도 해소해주신다.

## 피어 세션 정리

* DenseNet
  * Dense Connectivity : feature 정보 손상없이 쌓을 수 있다. 그레디언트 소실 문제도 개선된다.
  * Transition layer : pooling을 1x1 conv로 함

## 마스터 클래스

* 논문 추천 : [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

## 학습 회고

* 이번 주는 강의 내용을 따라가기에도 시간이 모자라서 심화학습을 많이 하지 못했다.
* 주말에 부족했던 개념 공부나 심화적인 내용을 보충해서 공부해야겠다.
* 다음 주는 파이토치를 배우니까 본격적으로 논문 구현 시도해보면 좋을 것 같다.
