# 부스트캠프 1일차

- [부스트캠프 1일차](#부스트캠프-1일차)
  - [체크인/체크아웃 루틴](#체크인체크아웃-루틴)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [MeetUP & 피어 세션 정리](#meetup--피어-세션-정리)
  - [강의 복습 내용](#강의-복습-내용)
    - [[1강] 벡터](#1강-벡터)
    - [[2강] 행렬](#2강-행렬)
    - [[3,4강] 경사하강법](#34강-경사하강법)
    - [[4강] 확률적 경사하강법(stochastic gradient descent)](#4강-확률적-경사하강법stochastic-gradient-descent)
  - [과제 수행 과정](#과제-수행-과정)
  - [학습 회고](#학습-회고)

## 체크인/체크아웃 루틴

* 체크인 : 오전 9시 30분~10시 / 체크아웃 : 오후 7시 이후
* 부스트코스 학습 시작 & 종료
* 부스트코스 본인 인증 -실명 정보 삭제 후 인증
* 슬랙 체크인 & 체크아웃

## 오늘 일정 정리

* 주제 : Python Basics, AI Math
* 8/2 (월)
  - [x] Python: 필수과제 1,2,3
  - [x] AI Math: 필수퀴즈 1~4강

## MeetUP & 피어 세션 정리

* 팀명 : 아29야(아직 2ㄴ공지능 초보지만 9래두 야 너두 할수 있어)
* 그라운드룰
  * 호칭 : ~님
  * 모더레이터 역할 : 회의록, 업로드, 회의 진행
* 협업툴 정리
  * 줌 : 회의
  * 노션 : 회의록

## 강의 복습 내용

### [1강] 벡터

* 용어
  * 벡터(vector) : 1차원 배열
  * 노름(norm) : 원점에서부터의 거리
    * <!-- $L_1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L_1">​-노름 : 각 성분의 변화량의 절대값을 모두 더함
    * <!-- $L_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L_2">​​-노름 : 유클리드 거리 계산
    * 노름의 종류에 따라 기하학적 성질이 달라진다.
  * 성분곱(Hadamard product) : 같은 모양을 가지는 벡터끼리의 곱
  * 내적(inner product)
    * <!-- $<x, y> = ||x||_2 ||y||_2 \cos \theta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%3Cx%2C%20y%3E%20%3D%20%7C%7Cx%7C%7C_2%20%7C%7Cy%7C%7C_2%20%5Ccos%20%5Ctheta">​​
    * 두 벡터의 유사도(similarity)를 측정하는데 사용 가능
* 두 벡터 사이의 거리 : 벡터의 뺄셈, 노름 이용
* 두 벡터 사이의 각도 : 내적(inner product), <!-- $L_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L_2">​-노름 이용

### [2강] 행렬

* 용어
  * 행렬(matrix) : 벡터를 원소로 가지는 2차원 배열
  * 전치행렬(transpose matrix) : 행과 열의 인덱스가 바뀐 행렬
  * 행렬의 덧셈, 뺄셈, 성분곱, 스칼라곱은 벡터와 차이가 없다.
  * 행렬 곱셈(matrix multiplication) : i번째 행벡터와 j번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산
  * 역행렬(inverse matrix) : 행과 열의 숫자가 같고 행렬식(determinant)이 0이 아닌 경우 구할 수 있다.
    * <!-- $A A^{-1} = A^{-1} A = I$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A%20A%5E%7B-1%7D%20%3D%20A%5E%7B-1%7D%20A%20%3D%20I">
* 유사역행렬(pseudo-inverse) 또는 무어-펜로즈(Moore-Penrose) 역행렬
  * <!-- $n \geq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Cgeq%20m">​ 인 경우 <!-- $A^{+} = (A^{T} A)^{-1} A^{T}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A%5E%7B%2B%7D%20%3D%20(A%5E%7BT%7D%20A)%5E%7B-1%7D%20A%5E%7BT%7D">, <!-- $A^{+}A = I$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A%5E%7B%2B%7DA%20%3D%20I">​
  * <!-- $n \leq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Cleq%20m"> 인 경우 <!-- $A^{+} = A^{T} (A A^{T})^{-1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=A%5E%7B%2B%7D%20%3D%20A%5E%7BT%7D%20(A%20A%5E%7BT%7D)%5E%7B-1%7D">, <!-- $AA^{+} = I$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=AA%5E%7B%2B%7D%20%3D%20I">
* 연립방정식 풀기 (<!-- $n \leq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Cleq%20m"> 인 경우)
  * <!-- $Ax = b \Rightarrow x = A^{+}b = A^{T} (AA^{T})^{-1}b$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Ax%20%3D%20b%20%5CRightarrow%20x%20%3D%20A%5E%7B%2B%7Db%20%3D%20A%5E%7BT%7D%20(AA%5E%7BT%7D)%5E%7B-1%7Db">​
  * 무어-펜로즈 역행렬을 이용하면 해를 하나 구할 수 있다.
* 선형회귀분석 (<!-- $n \geq m$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Cgeq%20m"> 인 경우)
  * <!-- $X \beta = \hat{y} \approx y \Rightarrow \beta = X^{+}y = (X^{T}X)^{-1}X^{T}y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X%20%5Cbeta%20%3D%20%5Chat%7By%7D%20%5Capprox%20y%20%5CRightarrow%20%5Cbeta%20%3D%20X%5E%7B%2B%7Dy%20%3D%20(X%5E%7BT%7DX)%5E%7B-1%7DX%5E%7BT%7Dy">​​​
  * (<!-- $\underset{\beta}{min} || y - \hat{y} ||_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cunderset%7B%5Cbeta%7D%7Bmin%7D%20%7C%7C%20y%20-%20%5Chat%7By%7D%20%7C%7C_2">, 즉, <!-- $L_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=L_2">​-노름을 최소화)
  * 선형회귀분석은 연립방정식과 달리 행이 더 크므로 방정식을 푸는건 불가능하고, <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y">에 근접하는 <!-- $\hat{y}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D">​​를 찾을 수 있다.

### [3,4강] 경사하강법

* 용어
  * 미분(differentiation) : 변화율의 극한(limit)
  * 편미분(partial differentiation) : 벡터가 입력인 다변수 함수일 때의 미분
  * 경사상승법(gradient ascent) : 미분값을 더함, 함수의 극댓값의 위치를 구할 때 사용
  * 경사하강법(gradient descent) : 미분값을 뺌, 함수의 극소값의 위치를 구할 때 사용
  * 그레디언트(gradient) 벡터 : <!-- $\nabla f = (\partial_{x_1}f, \partial_{x_2}f, ..., \partial_{x_d}f)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla%20f%20%3D%20(%5Cpartial_%7Bx_1%7Df%2C%20%5Cpartial_%7Bx_2%7Df%2C%20...%2C%20%5Cpartial_%7Bx_d%7Df)">
* 목표 : 선형회귀의 목적식을 최소화하는 <!-- $\beta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta">를 찾아야 한다.
  * 목적식이 <!-- $|| y - X \beta ||_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2">​ 일 때
    * 그레디언트 벡터 식
      * <!-- $\nabla_{\beta}|| y - X \beta ||_2 = (\partial_{\beta_1} || y - X \beta ||_2, ..., \partial_{\beta_d} || y - X \beta ||_2)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla_%7B%5Cbeta%7D%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%20%3D%20(%5Cpartial_%7B%5Cbeta_1%7D%20%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%2C%20...%2C%20%5Cpartial_%7B%5Cbeta_d%7D%20%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2)">​ 이고,
      * <!-- $\partial_{\beta_k} || y - X \beta ||_2 = \partial_{\beta_k}  \{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \sum_{j=1}^{d} X_{ij} \beta_{j})^2 \}^{1/2} = - \frac{X^T_k (y - X \beta)}{n || y - X \beta ||_2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cpartial_%7B%5Cbeta_k%7D%20%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%20%3D%20%5Cpartial_%7B%5Cbeta_k%7D%20%20%5C%7B%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20(y_i%20-%20%5Csum_%7Bj%3D1%7D%5E%7Bd%7D%20X_%7Bij%7D%20%5Cbeta_%7Bj%7D)%5E2%20%5C%7D%5E%7B1%2F2%7D%20%3D%20-%20%5Cfrac%7BX%5ET_k%20(y%20-%20X%20%5Cbeta)%7D%7Bn%20%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%7D">​ 이므로,​​
      * <!-- $\nabla_{\beta}|| y - X \beta ||_2 = - \frac{X^T (y - X \beta)}{n || y - X \beta ||_2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla_%7B%5Cbeta%7D%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%20%3D%20-%20%5Cfrac%7BX%5ET%20(y%20-%20X%20%5Cbeta)%7D%7Bn%20%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%7D">​​
      * 복잡한 계산이지만 사실 <!-- $X \beta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X%20%5Cbeta">​를 계수 <!-- $\beta$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta">​에 대해 미분한 결과인 <!-- $X^T$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X%5ET">​​만 곱해지는 것
    * 경사하강법 알고리즘 : <!-- $\beta^{(t+1)} \leftarrow \beta^{(t)} - \lambda \nabla_{\beta}|| y - X \beta^{(t)} ||$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta%5E%7B(t%2B1)%7D%20%5Cleftarrow%20%5Cbeta%5E%7B(t)%7D%20-%20%5Clambda%20%5Cnabla_%7B%5Cbeta%7D%7C%7C%20y%20-%20X%20%5Cbeta%5E%7B(t)%7D%20%7C%7C">
  * 목적식이 <!-- $|| y - X \beta ||_2^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%5E2">​ 일 때
    * 그레디언트 벡터 식 : <!-- $\nabla_{\beta}|| y - X \beta ||_2^2 = - \frac{2}{n} X^T (y - X \beta)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla_%7B%5Cbeta%7D%7C%7C%20y%20-%20X%20%5Cbeta%20%7C%7C_2%5E2%20%3D%20-%20%5Cfrac%7B2%7D%7Bn%7D%20X%5ET%20(y%20-%20X%20%5Cbeta)">
    * 경사하강법 알고리즘 : <!-- $\beta^{(t+1)} \leftarrow \beta^{(t)} + \frac{2 \lambda}{n} X^T (y - X \beta^{(t)})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta%5E%7B(t%2B1)%7D%20%5Cleftarrow%20%5Cbeta%5E%7B(t)%7D%20%2B%20%5Cfrac%7B2%20%5Clambda%7D%7Bn%7D%20X%5ET%20(y%20-%20X%20%5Cbeta%5E%7B(t)%7D)">​
* 특징
  * 학습률(lr)과 학습횟수(epoch)가 중요한 하이퍼파라미터(hyperparameter)
  * convex한 함수에 대해서는 수렴 보장

### [4강] 확률적 경사하강법(stochastic gradient descent)

* 특징
  * 일부 데이터를 활용하여 업데이트
  * 연산자원을 효율적으로 활용 가능 (메모리 등 하드웨어 한계 극복 가능)
  * **non-convex 목적식을 최적화할 수 있다. (머신러닝 학습에 더 효율적)**
* 원리 : 미니배치 연산
  * **목적식이 미니배치마다 조금씩 달라진다.**
  * 따라서 극소점/극대점이 바뀔 수 있다. (극소점에서 탈출할 수 있다.)


## 과제 수행 과정

* Assignment1 : numpy 이용
* Assignment2,3 : re 이용

## 학습 회고

* 생각보다 오래 앉아있는게 힘들다.
* 앞으로 부캠 공부와 개인 공부를 병행할 수 있을지 모르겠다.
