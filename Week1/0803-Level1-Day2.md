# 부스트캠프 2일차

- [부스트캠프 2일차](#부스트캠프-2일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 복습 내용](#강의-복습-내용)
    - [[5강] 딥러닝 학습방법](#5강-딥러닝-학습방법)
    - [[6강] 확률론](#6강-확률론)
  - [피어 세션 정리](#피어-세션-정리)
  - [과제 수행 과정](#과제-수행-과정)
    - [※ 정규표현식 사용법 정리](#-정규표현식-사용법-정리)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/3 (화)
  - [x] Python: 필수과제 4,5
  - [x] AI Math: 필수퀴즈 5~6강
* 오늘, 내일까지 모더레이터 담당!

## 강의 복습 내용

### [5강] 딥러닝 학습방법

* 소프트맥스(softmax) 함수
  * 모델의 출력을 확률로 해석할 수 있게 변환해주는 연산
  * 학습할 때 사용되고, 보통 추론하는 경우에는 사용하지 않는다.

* 활성함수(activation function)
  * 비선형함수(nonlinear function)
  * 활성함수가 없으면 선형모형과 차이가 없다.
  * 소프트맥스 함수와는 달리, 계산할 때 그 자리의 실수값만 고려한다.
  * 종류 : sigmoid, tanh, ReLU 등

* 신경망 : 선형모델과 활성함수를 합성한 함수
  * 다층퍼셉트론(MLP) : 신경망이 여러층 합성된 함수
  * universal approximation theorem : 이론적으로 2층 신경망으로 임의의 연속함수를 근사할 수 있다.
  * 층이 깊으면 더 적은 노드로 목적함수 근사가 가능하다.

* 순전파(forward propagation) : 레이어의 순차적인 신경망 계산
* 역전파(backpropagation) : 출력에서부터 역순으로 계산
  * 원리 : 연쇄법칙(chain-rule) 기반 자동미분(auto-differentiation)
  * 특징 : 순전파 결과와 미분값을 둘 다 저장해야 한다.

### [6강] 확률론

* 딥러닝은 확률론 기반의 기계학습 이론 바탕 (손실함수 작동원리)

* 확률변수 : 관측가능한 데이터, 함수로 해석
  * 확률변수 구분 : (데이터 공간이 아니라) 확률분포에 의해 결정
  * 이산확률변수(discrete) : 확률변수가 가질 수 있는 모든 경우의 수의 확률을 더해서 모델링
    * 확률질량함수 : <!-- $P(X \in A) = \sum_{x \in A} P(X=x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(X%20%5Cin%20A)%20%3D%20%5Csum_%7Bx%20%5Cin%20A%7D%20P(X%3Dx)">
  * 연속확률변수(continuous) : 데이터 공간에 정의된 확률변수의 밀도(density) 위에서 적분을 통해 누적확률분포의 변화율을 모델링
    * 밀도함수 :  <!-- $P(X \in A) = \int_A P(x) dx$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(X%20%5Cin%20A)%20%3D%20%5Cint_A%20P(x)%20dx">

* 확률분포 : 데이터를 표현하는 초상화, 기계학습을 통해 확률분포 추론
  * 결합확률분포 <!-- $P(x,y)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(x%2Cy)">
    * 주어진 데이터의 결합분포 <!-- $P(x,y)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(x%2Cy)">를 이용하여 원래 확률분포 <!-- $D$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=D"> 모델링
  * 주변확률분포 <!-- $P(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(x)"> : 입력 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">에 대한 정보
    * <!-- $P(x) = \sum_y P(x, y)$ or $P(x) = \int_y P(x,y) dy$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(x)%20%3D%20%5Csum_y%20P(x%2C%20y)%24%20or%20%24P(x)%20%3D%20%5Cint_y%20P(x%2Cy)%20dy">
  * 조건확률분포 <!-- $P(x|y)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(x%7Cy)">​ : 특정 클래스일 때의 데이터의 확률분포
    * 데이터 공간에서 입력 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">와 출력 <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y">​ 사이의 관계 모델링
  * 조건부확률 <!-- $P(y|x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P(y%7Cx)"> : 입력변수 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">​에 대해 정답이 <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y">​​​일 확률(분류 문제)
    * 선형모델과 소프트맥스 함수의 결합 등 
  * 조건부기대값 <!-- $E[y|x]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E%5By%7Cx%5D"> : 입력변수 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">에 대해 정답이 <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y">​일 밀도(회귀 문제)
    * <!-- $E_{y \sim P(y|x)}[y|x] = \int_y y P(y|x) dy$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E_%7By%20%5Csim%20P(y%7Cx)%7D%5By%7Cx%5D%20%3D%20%5Cint_y%20y%20P(y%7Cx)%20dy">
    * <!-- $E||y - f(x)||_2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=E%7C%7Cy%20-%20f(x)%7C%7C_2"> (L2-노름)을 최소화하는 함수 <!-- $f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)">와 일치 (수학적으로 증명됌)

* 통계적 범함수(statistical functional) : 확률분포에서 데이터를 분석하는데 사용
  * 기대값(expectation) : 데이터를 대표하는 통계량, 평균(mean)
  * 분산(variance), 첨도(skewness), 공분산(covariance) 등 : 기대값을 통해 계산 가능

* 몬테카를로(Monte Carlo) 샘플링 : 확률분포를 명시적으로 모를 때, 데이터를 이용하여 기대값 계산
  * 독립추출이 보장되면, 대수의 법칙(law of large number)에 의해 수렴성 보장
  * 부정적분 방식으로는 어려웠던 적분을 계산할 수 있다.
  * 적절한 샘플사이즈로 계산해야 오차범위를 줄일 수 있다.

## 피어 세션 정리

* 협업 툴 및 피어세션 계획 점검
  * 과제 코드 리뷰를 구글 드라이브에서 하기로 결정
  * 피어세션 전에 과제 수행여부 체크 후에 과제 파일을 업로드하여 각각 한명씩 맡아서 코멘트 달고 의견 공유하기
* 강의 내용 질문 및 심화 내용 정리 공유
  * Q. Assignment3의 tocamelcase 함수에서, for문을 직접 쓰지 않고 풀 수도 있을지
    * `reduce, lambda`로 구현 가능
  * Q. 무어-펜로즈 역행렬을 이용해서 선형회귀분석 할 수 있다고 하는데, y절편을 직접 더해줘야 한다는 말이 뭔지 모르겠다.
    * 선형회귀의 bias term 이다.
* 과제 관련 질문 및 아이디어 공유
  * 오늘 과제는 출력 부분을 신경써야 해서 조금 까다로웠다.

## 과제 수행 과정

* Assignment4 : main 출력에서 자꾸 에러가 났는데, while문 강제종료 조건과 게임오버 조건을 추가함으로써 해결할 수 있었다.
* Assignment5 : 정규표현식 쓸 때, 특수문자가 있는지 확인하는 작업이 어려웠다. 정규표현식에 사용되는 문자 앞에 역슬래시를 넣어서 해결하였다.

### ※ 정규표현식 사용법 정리

* 메소드 (참고 : [정규표현식 re 라이브러리 문서](https://docs.python.org/ko/3/library/re.html#module-contents))
  * `re.search(pattern, string)` : string 전체를 검색하여 정규식 pattern과 일치하는 첫번째 위치를 찾는다. (match object 또는 None 반환)
  * `re.sub(pattern, repl, string)` : string에서 pattern과 일치하는 곳을 repl로 치환하여 얻은 문자열을 반환한다. 패턴을 찾지 못하면 string 그대로 반환된다.
* 정규식 문법 (참고 : [정규식 HOWTO](https://docs.python.org/ko/3/howto/regex.html#regex-howto))
  * `[`와 `]` : 일치시키려는 문자 집합인 문자 클래스를 지정하는데 사용
  * `-` : 문자의 범위 나타내기 (`[a-z]`는 소문자 전체)
  * `^` : 여집합 나타내기 (`[^a-z]`는 소문자 제외)
  * `\` : 모든 메타 문자 이스케이프 처리, 특수 시퀀스 나타내기
    * `\d`는 모든 십진 숫자(=`[0-9]`), `\D`는 모든 비 숫자 문자(=`[^0-9]`)
    * `\w`는 모든 영숫자(=`[a-zA-Z0-9_]`), `\W`는 모든 비 영숫자(=`[^a-zA-Z0-9_]`)
    * `\s`는 모든 공백 문자(=`[\t\n\r\f\v]`), `\S`는 모든 비 공백 문자(=`[^\t\n\r\f\v]`)
  * `*` : 0개 이상과 일치, `+` : 1개 이상과 일치

## 학습 회고

* 어제 피곤해서 일찍 자는 바람에 오늘 몰아 듣느라 힘들었다.
* 내일은 제때 듣고, 정리도 미리미리 해야겠다.
* 자연어 처리 공부를 하고 싶은데, 수학 공부를 먼저 해야하나 생각이 들었다.
