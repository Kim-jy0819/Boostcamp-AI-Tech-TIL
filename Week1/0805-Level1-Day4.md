# 부스트캠프 4일차

## 오늘 일정 정리

* 8/5 (목)
  - [x] AI Math: 필수퀴즈 10강
  - [x] Python: 선택과제 2,3
  - [x] 마스터클래스 (임성빈 교수님) (8/5(목) 18:00~19:00) : AI & Math FAQ
  - [x] 29조 멘토링 오후 8시~

## 강의 복습 내용

### [10강] RNN

* 시퀀스(sequence) 데이터 : 소리, 문자열, 주가 등

* 시퀀스 데이터를 다룰 수 있는 모델
  1. 조건부 확률 이용(베이즈 법칙)
     * <!-- $P(X_1, ..., X_t) = P(X_t | X_1, ..., X_{t-1}) P(X_1, ..., X_{t-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(X_1%2C%20...%2C%20X_t)%20%3D%20P(X_t%20%7C%20X_1%2C%20...%2C%20X_%7Bt-1%7D)%20P(X_1%2C%20...%2C%20X_%7Bt-1%7D)">
     * <!-- $X_t \sim P(X_t | X_{t-1}, ..., X_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_t%20%5Csim%20P(X_t%20%7C%20X_%7Bt-1%7D%2C%20...%2C%20X_1)">​
     * <!-- $X_{t+1} \sim P(X_{t+1} | X_t, X_{t-1}, ..., X_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_%7Bt%2B1%7D%20%5Csim%20P(X_%7Bt%2B1%7D%20%7C%20X_t%2C%20X_%7Bt-1%7D%2C%20...%2C%20X_1)">
  2. <!-- $AR(\tau)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=AR(%5Ctau)"> (Autoregressive Model) : 자기회귀모델, 고정된 길이 <!-- $\tau$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctau">​​ 만큼의 시퀀스만 사용
  3. 잠재 AR 모델 : 바로 이전 정보 <!-- $X_{t-1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_%7Bt-1%7D">​​를 제외한 나머지 정보들 <!-- $X_{t-2}, ..., X_1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_%7Bt-2%7D%2C%20...%2C%20X_1">​​ 을 <!-- $H_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t">​​​라는 잠재변수로 인코딩에서 활용하는 방법
  4. RNN 모델 : 잠재변수 $H_t$​​를 신경망을 통해 반복해서 사용하여 시퀀스 데이터의 패턴을 학습하는 모델
     * <!-- $X_t \sim P(X_t | X_{t-1}, H_t)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_t%20%5Csim%20P(X_t%20%7C%20X_%7Bt-1%7D%2C%20H_t)">​, <!-- $H_t = Net_{\theta} (H_{t-1}, X_{t-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t%20%3D%20Net_%7B%5Ctheta%7D%20(H_%7Bt-1%7D%2C%20X_%7Bt-1%7D)">​
     * <!-- $X_{t+1} \sim P(X_{t+1} | X_t, H_{t+1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X_%7Bt%2B1%7D%20%5Csim%20P(X_%7Bt%2B1%7D%20%7C%20X_t%2C%20H_%7Bt%2B1%7D)">

* forward
  * MLP
    * <!-- $H_t = \sigma (X_t W^{(1)} + b^{(1)})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t%20%3D%20%5Csigma%20(X_t%20W%5E%7B(1)%7D%20%2B%20b%5E%7B(1)%7D)">
    * <!-- $O_t = H_t W^{(2)} + b^{(2)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O_t%20%3D%20H_t%20W%5E%7B(2)%7D%20%2B%20b%5E%7B(2)%7D">
  * RNN : MLP와 유사한 모양이다.
    * <!-- $H_t = \sigma (X_t W^{(1)}_X + H_{t-1} W^{(1)}_H + b^{(1)})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_t%20%3D%20%5Csigma%20(X_t%20W%5E%7B(1)%7D_X%20%2B%20H_%7Bt-1%7D%20W%5E%7B(1)%7D_H%20%2B%20b%5E%7B(1)%7D)">​ ( <!-- $H_{t-1} W^{(1)}_H$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_%7Bt-1%7D%20W%5E%7B(1)%7D_H">​ term 추가)
    * <!-- $O_t = H_t W^{(2)} + b^{(2)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O_t%20%3D%20H_t%20W%5E%7B(2)%7D%20%2B%20b%5E%7B(2)%7D">​
    * <!-- $H_{t+1} = \sigma (X_{t+1} W^{(1)}_X + H_{t} W^{(1)}_H + b^{(1)})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=H_%7Bt%2B1%7D%20%3D%20%5Csigma%20(X_%7Bt%2B1%7D%20W%5E%7B(1)%7D_X%20%2B%20H_%7Bt%7D%20W%5E%7B(1)%7D_H%20%2B%20b%5E%7B(1)%7D)">​
    * <!-- $O_{t+1} = H_{t+1} W^{(2)} + b^{(2)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=O_%7Bt%2B1%7D%20%3D%20H_%7Bt%2B1%7D%20W%5E%7B(2)%7D%20%2B%20b%5E%7B(2)%7D">​

* backward
  * BPTT(Backpropagation Through Time) : RNN의 역전파 방법
    * 잠재변수의 연결그래프에 따라 순차적으로 계산
    * 미분의 곱으로 이루어진 항 계산 : 시퀀스 길이가 길어질수록 이 항은 불안정해지기 쉽다. (vanishing gradient)
  * truncated BPTT : 시퀀스 길이가 길어지는 경우 역전파 알고리즘 계산이 불안정해지므로 길이를 끊는다.

* LSTM, GPU : 긴 시퀀스를 처리하기 위해 등장한 네트워크

## 피어 세션 정리

* 과제 코드 리뷰 후 배운 점
  * 비슷한 변수가 많아서 헷갈릴 수 있는데 이름을 잘 지어주셔서 보기 좋았다.
  * 정규표현식을 좀 더 익숙하게 쓰고 싶다.
  * any, all, map, reversed 등 파이썬의 여러 기능을 사용할 수도 있다는 것을 깨달았다.

* 강의 질문
  * Q. 쿨백-라이블러 발산이 항상 0보다 크거나 같은 이유? 어떻게 <!-- $KL(P||Q) = \int_{X} P(x) \log (\frac{P(x)}{Q(x)}) dx$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=KL(P%7C%7CQ)%20%3D%20%5Cint_%7BX%7D%20P(x)%20%5Clog%20(%5Cfrac%7BP(x)%7D%7BQ(x)%7D)%20dx"> 이 식이 항상 <!-- $KL(P||Q) \geq 0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=KL(P%7C%7CQ)%20%5Cgeq%200"> 을 만족할 수 있는 것인지 궁금
  * 답변) https://hyunw.kim/blog/2017/10/27/KL_divergence.html

## 과제 수행 과정

* 선택과제2
  * y가 마지막 상태에서 1의 개수를 세는 건데, 결국 <!-- $s_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=s_n"> 이 1이면 1개이므로 y를 <!-- $s_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=s_n">로 loss를 계산해도 괜찮다.
  * gradient 구할 때, <!-- $w_x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_x">가 <!-- $s_1, ..., s_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=s_1%2C%20...%2C%20s_n">을 만드는데에 관여했으므로 각각에 체인룰을 적용하여 구한 gradient의 합을 구한다. <!-- $w_{rec}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_%7Brec%7D">​​ 도 마찬가지로 구한다.

* 선택과제3
  * 변수의 개수 n과 표본의 개수 n을 다르게 표기했어야 하는거 아닌가 싶긴 하다.
  * 수식을 직접 손으로 써서 계산하니까 이해가 잘 되었다.
  * 도대체 무슨 그래프를 그리는건지 처음에는 이해가 잘 안갔는데, 서로 다른 mu를 가진 정규분포의 확률 밀도 함수를 그리는 것이었다.
  * 샘플에서의 확률밀도함수 값의 합을 구해야하는데 샘플이 하나여서 결국 그래프가 확률밀도함수가 되는 것 같다.

## 마스터 클래스

* 인공지능 수학
  * 어떻게 공부?
    * 용어의 정의를 외우는 것부터 시작 : 교과서나 위키피디아
    * 용어를 외운 후에는 예제를 찾아보자 : 책보단 구글링
  * 어느 정도로 공부?
    * 원리를 이해하기 위한 기초, 필요한걸 공부해서 빠르게 따라잡을 수 있을만큼 알아야 한다.
  * 어떤 걸 공부?
    * 선형대수 / 확률론 / 통계학
    * Dive into Deep Learning : 딥러닝 책인데, appendix에 mathematics for machine learning 섹션이 있다.
  * 언제 쓰이는지?
    * 문제를 정의하고 브레인스토밍하는 부분에서 특히 필요하다.

* 인공지능 대학원
  * 학석박 간의 차이
    * 대중화되지 않은 영역은 학위과정이 중요할 수 있다.
  * 중요한 요소
    1. 자기가 쓴 논문 있는지
    2. 코드가 공개 안 된 논문 구현 및 오픈소스로 공개
    3. 대학원은 인턴 연계 잘 되어서 채용 시 레퍼런스 체크하기 좋아서 선호하는 경향있음

## 멘토링

* 워라벨 회사마다 너무 다르다. 딥러닝 연구자는 워라벨 좋은 편이다.
* 논문 구현 : classification - vggnet, inception, resnet, mobilenet, googlenet
  * https://www.youtube.com/channel/UClZH5RmZxu_9NMz7_hDRgTQ
  * 토이데이터셋 : mnist, cifar-10 (32 by 32), cifar-100, imagenet은 너무 큼
    * 블록을 조금만 쌓기 or 이미지 사이즈를 늘리고 하기
* 코드 리뷰할 때 : ipynb 대신 py 파일로 하는 연습하기, 깃허브에서의 코드 관리도 편하다. 깃 쓰는 연습 하는게 좋은 것 같다.
* 부스트캠프 수료 : 주니어 엔지니어(구현)/리서쳐(연구) 둘 다 할 수 있다.
* 수학공부는 지금 공부할 수 있으면 미리하는게 좋은 것 같다.

## 학습 회고

* 수학은 쓰면서 공부하자
* 금요일에 파이썬 복습 빨리 하고 Dive into Deeplearning 수학 공부 시작하면 좋겠다.
