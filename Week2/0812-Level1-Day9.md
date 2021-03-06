# 부스트캠프 9일차

- [부스트캠프 9일차](#부스트캠프-9일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[7강] Sequential Models - RNN](#7강-sequential-models---rnn)
    - [[8강] Sequential Models - Transformer](#8강-sequential-models---transformer)
  - [과제 수행 과정](#과제-수행-과정)
  - [피어 세션 정리](#피어-세션-정리)
  - [특강 내용 정리](#특강-내용-정리)
  - [오피스 아워](#오피스-아워)
  - [멘토링](#멘토링)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/12 (목)
  - [x] DL Basic
    - [x] (7강) Sequential models - RNN
    - [x] [필수 과제4] LSTM Assignment
    - [x] (8강) Sequential models - Transformer
    - [x] [필수 과제5] Multi-headed Attention Assignment
  - [x] 이고잉님의 Git/Github 특강 13:00~15:30
  - [x] 오피스아워 18:00~19:30 - 과제 해설
  - [x] 멘토링 8시~

## 강의 내용 정리

### [7강] Sequential Models - RNN

* Sequential Data
  * 오디오, 비디오 등
  * 입력의 차원을 알 수 없어서 처리하는데에 어려움
  * 몇 개의 입력이 들어오는지에 상관없이 모델은 동작해야 함

* Naive Sequential Model
  * 이전의 입력에 대해서 다음에 어떤 출력이 나올지 예측
  * <!-- $p(x_t | x_{t-1}, x_{t-2}, ...)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_t%20%7C%20x_%7Bt-1%7D%2C%20x_%7Bt-2%7D%2C%20...)">​​
    * <!-- $x_{t-1}, x_{t-2}, ...$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bt-1%7D%2C%20x_%7Bt-2%7D%2C%20...">​ : The number of inputs varies, 고려해야하는 과거의 정보가 점점 늘어남
  * <!-- $p(x_t | x_{t-1}, ..., x_{t-r})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_t%20%7C%20x_%7Bt-1%7D%2C%20...%2C%20x_%7Bt-r%7D)">
    * <!-- $x_{t-r}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bt-r%7D"> : Fix the past timespan, 과거의 r 개의 정보만 고려한다.

* Markov model (first-order autogressive model)
  * <!-- $p(x_1,...,x_T) = p(x_T|x_{T-1}) p(x_{T-1}|x_{T-2}) ... p(x_2|x_1) p(x_1) = \Pi_{t=1}^{T} p(x_t | x_{t-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p(x_1%2C...%2Cx_T)%20%3D%20p(x_T%7Cx_%7BT-1%7D)%20p(x_%7BT-1%7D%7Cx_%7BT-2%7D)%20...%20p(x_2%7Cx_1)%20p(x_1)%20%3D%20%5CPi_%7Bt%3D1%7D%5E%7BT%7D%20p(x_t%20%7C%20x_%7Bt-1%7D)">
  * 현재는 (바로 전) 과거에만 의존적이다.
  * 과거의 많은 정보를 버리는 것이 됨

* Latent autogressive model 
  * <!-- $\hat{x} = p(x_t|h_t)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%7Bx%7D%20%3D%20p(x_t%7Ch_t)">
    * <!-- $h_t = g(h_{t-1}, x_{t-1})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_t%20%3D%20g(h_%7Bt-1%7D%2C%20x_%7Bt-1%7D)">
    * <!-- $h_{t-1}, h_t$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_%7Bt-1%7D%2C%20h_t"> : summary of the past
  * 현재는 바로 전 과거 하나가 아니라, 이전의 정보를 요약하는 hidden state에 의존적이다.

* RNN(Recurrent Neural Network)
  * 앞서 나온 내용을 가장 쉽게 구현하는 방법
  * ![image](https://user-images.githubusercontent.com/35680202/129124658-70742af7-2c9d-4881-8e3e-c97e643a1a0d.png)
  * 단점
    * Long-term dependencies를 잡는 것이 어렵다.
      * 먼 과거에 있는 정보가 미래에 영향을 주기까지 남아있기가 어렵다.
    * 학습이 어렵다.
      * <!-- $h_1 = \Phi(W^T h_0 + U^T x_1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_1%20%3D%20%5CPhi(W%5ET%20h_0%20%2B%20U%5ET%20x_1)">
      * ...
      * <!-- $h4 = \Phi(W^T \Phi(W^T \Phi(W^T \Phi(W^T h_0 + U^T x_1) + U^T x_2) + U^T x_3) + U^T x_4)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h4%20%3D%20%5CPhi(W%5ET%20%5CPhi(W%5ET%20%5CPhi(W%5ET%20%5CPhi(W%5ET%20h_0%20%2B%20U%5ET%20x_1)%20%2B%20U%5ET%20x_2)%20%2B%20U%5ET%20x_3)%20%2B%20U%5ET%20x_4)">
        * ex) activation function이 sigmoid인 경우 : vanishing gradient
        * ex) activation function이 relu인 경우 : exploding gradient

* LSTM(Long Short Term Memory)
  * 구조
    * Forget gate : Decide which information to throw away
      * <!-- $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f_t%20%3D%20%5Csigma(W_f%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_f)">
    * Input gate : Decide which information to store in the cell state
      * <!-- $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=i_t%20%3D%20%5Csigma(W_i%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_i)">
    * Update cell
      * <!-- $\tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7BC_t%7D%20%3D%20%5Ctanh(W_C%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_C)"> : 예비 cell state
      * <!-- $C_t = f_t * C_{t-1} + i_t * \tilde{C_t}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=C_t%20%3D%20f_t%20*%20C_%7Bt-1%7D%20%2B%20i_t%20*%20%5Ctilde%7BC_t%7D"> : cell state (timestep t 까지 들어온 정보 요약)
    * Output gate : Make output using the updated cell state
      * <!-- $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=o_t%20%3D%20%5Csigma(W_o%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D%20%2B%20b_o)">
      * <!-- $h_t = o_t * \tanh(C_t)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_t%20%3D%20o_t%20*%20%5Ctanh(C_t)">

* GRU(Gated Recurrent Unit)
  * 구조
    * Update gate
      * <!-- $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=z_t%20%3D%20%5Csigma(W_z%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D)">​
    * Reset gate
      * <!-- $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=r_t%20%3D%20%5Csigma(W_r%20%5Ccdot%20%5Bh_%7Bt-1%7D%2C%20x_t%5D)">​
    * Update hidden state
      * <!-- $\tilde{h_t} = \tanh(W \cdot [r_t * h_{t-1}, x_t])$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7Bh_t%7D%20%3D%20%5Ctanh(W%20%5Ccdot%20%5Br_t%20*%20h_%7Bt-1%7D%2C%20x_t%5D)">
      * <!-- $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h_t}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_t%20%3D%20(1%20-%20z_t)%20*%20h_%7Bt-1%7D%20%2B%20z_t%20*%20%5Ctilde%7Bh_t%7D">
  * 핵심
    * No cell state, just hidden state

### [8강] Sequential Models - Transformer

* **Transformer** is the first sequence transduction model based entirely on **attention**.

* Sequence to Sequence : NMT(Neural Machine Translation) 등
  * 입력시퀀스와 출력시퀀스의 단어의 개수가 다를 수 있다.
  * 입력시퀀스의 도메인과 출력시퀀스의 도메인이 다를 수 있다.
  * 근데 모델은 하나의 모델로 학습해야 함

* Transformer
  * **Encoder - Decoder** 구조 핵심
    * n개의 단어가 어떻게 Encoder에서 한번에 처리되는가
    * Encoder와 Decoder 사이에 어떤 정보가 주고 받아지는지
    * Decoder가 어떻게 generation할 수 있는지
  * 특징
    * CNN, MLP와 달리, 입력이 고정되더라도 옆에 주어진 다른 입력들이 달라짐에 따라서 출력이 달라질 수 있는 여지가 있다.
    * 따라서 훨씬 더 flexable하고 더 많은 것을 표현할 수 있다.
    * 하지만 계산량은 한번에 계산하기 때문에, (입력의 길이)^2이다.

* Encoder : **Self-Attention** + Feed-forward Neural Network
  * Feed-forward Neural Network : word-independent and parallelized
  * **Self-Attention**
    * 각 단어의 임베딩 벡터를 3가지 벡터(Queries(Q), Keys(K), Values(V))로 encoding 한다. - 몇 차원으로 할지는 hyperparameter
    * x1이 z1으로 변환될 때, 단순히 x1만 보는 것이 아니라 x2, x3도 같이 본다.
    * 따라서 Self-Attention은 dependencies가 있다.
  * 인코딩 과정
    * <img src="https://user-images.githubusercontent.com/35680202/129227050-b686786e-9872-4628-bcf4-ed656f6b823b.png" width="400" height="400">
    * Embedding vector : 단어를 임베딩한 벡터
    * **Query vector, Key vector, Value vector** : 각각의 neural network를 통해서 두 벡터를 만든다. (Q와 K는 내적해야하기 때문에 항상 차원이 같아야 한다.)
      * <img src="https://user-images.githubusercontent.com/35680202/129230769-be6854a4-f289-4724-91e7-c11961352222.png" width="300" height="300">
      * 행렬을 활용하여 한번에 찾을 수 있다.
      * <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X">​​​ : 두 단어에 대한 4차원 임베딩 벡터
      * <!-- $W^Q, W^K, W^V$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W%5EQ%2C%20W%5EK%2C%20W%5EV">를 찾는 multi-layer perceptron이 있다.
      * 이 multi-layer perceptron은 인코딩되는 단어들마다 다 shared 된다.
    * Score vector : i번째 단어에 대한 Score vector를 계산할 때, i번째 단어의 Query vector와 자기 자신을 포함한 나머지 단어들의 Key vector들을 내적한다.
      * 두 벡터가 얼마나 align이 잘 되어있는지 본다.
      * i번째 단어가 자기 자신을 포함한 나머지 단어들과 얼마나 관계가 있는지를 나타냄 (이것이 결국 attention)
      * 그 후 Score vector를 normalize해주고 softmax를 취해준다.
    * 최종 결과물 : Value vector 와 weighted sum을 해서 만든 인코딩 된 벡터
      * <img src="https://user-images.githubusercontent.com/35680202/129231629-e6b8b9e2-f8c9-4ade-9d77-971e08f2e7c6.png" width="350" height="150">
      * Value vector의 weight를 구하는 과정이 각 단어에서 Query vector와 Key vector 사이의 내적을 normalize&softmax 취해주고 나오는 attention을 Value vector와 weighted sum을 한 것이 최종적으로 나오는 인코딩된 벡터
      * 여기서는 인코딩된 벡터가 Value vector의 차원과 같다.
  * **Multi-headed attention(MHA)**
    * attention을 여러 번 하는 것
    * 즉, 하나의 임베딩된 벡터에 대해서 Query, Key, Value를 하나만 만드는 것이 아니라 N개 만드는 것
    * 따라서 하나의 임베딩된 벡터가 있으면 N개의 인코딩된 벡터를 얻을 수 있다.
    * 그 다음 인코더에 넣을 때 차원을 맞춰주기 위해서 다시 행렬을 곱해준다.
  * 최종 그림
    * <img src="https://user-images.githubusercontent.com/35680202/129234393-0f693f3d-e621-4462-9bd4-230ecdcb169c.png" width="700" height="400">
    * 사실 실제 구현이 이렇지는 않다.
    * Embedding dimension이 100이고, 10개의 head를 사용한다고 하면, 100 dimension을 10개로 나눠서 각각 10 dimension짜리를 가지고 Q, K, V를 만든다.
  * Positional encoding
    * 입력에 특정 값을 더해주는 것 (bias라고 보면 됨)
    * Transformer구조가 N개의 단어의 Sequential한 정보가 사실 포함되어 있지 않기 때문에 사용함
    * Positional encoding은 pre-defined된 특정 방법으로 벡터를 만든다.

* Decoder
  * Self-Attention
    * Decoder의 self-attention layer 에서 이전 단어들만 dependent하고 뒤(미래)에 있는 단어들은 dependent하지 않게(활용하지 않게) 만들기 위해 마스킹한다.
  * Encoder-Decoder Attention
    * Encoder에서 Decoder로 Key vector와 Value vector를 보낸다.
    * 이전 레이어의 Query vector와 Encoder에서 받아온 Key vector와 Value vector들을 가지고 최종 값이 나오게 된다.
  * Final layer
    * 단어들의 분포를 만들어서 그 중의 단어 하나를 매번 sampling 하는 식으로 동작
    * 출력은 Autoregressive 하게 하나의 단어씩 만든다. (I 가 들어가면 am 이 출력되는)

* Vision Transformer(ViT)
  * 이미지 분류를 할 때 Transformer Encoder만 활용한다.
  * Encoder에서 나온 첫번째 encoded vector를 classifier에 집어넣는 방식
  * 이미지에 맞게 하기 위해서 이미지를 영역으로 나누고 서브패치들을 linear layer를 통과해서 그게 하나의 입력인 것처럼 넣어준다. (positional embedding 포함)

* [DALL-E](https://openai.com/blog/dall-e/)
  * 문장에 대한 이미지를 만들어낸다.

## 과제 수행 과정

* 필수과제4 - LSTM
* 필수과제5 - Transformer

## 피어 세션 정리

* Inception-v2
  * 최적화 도구들 몇개 적용해서 성능을 개선함
  * More Factorization : factorizing inception module
  * Spatial Factorization into Asymmetric Convolutions
  * 맨 앞단 auxiliary classifier 제거
  * Efficient Grid Size Reduction
    * Pooling을 해줘야하는 이유 : parameter 증가 없이 downsizing으로 overfitting 방지
    * 순서에 대한 논의 : 정보 손실 vs 연산량 증가

* Inception-v3
  * Label Smoothing
    * 모델의 예측을 너무 확신하는 것을 줄여보자.
    * ex) one-hot vector를 1을 0.9로 줄이고, 0을 0.1로 숫자로 좀 더 쪼개서 분류
    * 하이퍼파라미터 입실론 이용
  * Optimizer 종류를 바꿔보고, 결국 RMSProp을 사용하였다.
  * auxiliary classifier에도 Batch norm을 썼다.
  * 위의 모든 기법들을 싹 다 적용한 것이 Inception-v3

* XceptionNet
  * Inception model 에서 변형한 것
  * Inception-v3를 간단하게 나타내자
  * (1) extreme version of inception module
  * (2) Depthwise Separable Convolution
  * Xception architecture
    * residual connection 추가
    * resnet, inception-v3보다는 성능이 좋다.
    * 간결하게 코드를 짤 수 있다.

## 특강 내용 정리

* 목표
  * 오늘은 로컬에서 저장소를 만들고 원격으로 올리는 것
  * 원격 저장소가 없어도 로컬에서 버전관리 가능

* 복수 개의 작업의 커밋을 나누는 방법
  * Stage area : 커밋 대기 상태
  * Stage changes : 변경사항을 선택해서 커밋 대기 상태로 만들 수 있다. (git add)

* 버그
  * 문법 에러 : 가장 잡기 쉬운 에러
  * 논리 에러 : 잡기 어려운 에러

* 버전 관리의 본질
  * 버전 관리를 안하고 있다면 버그를 찾기 위해 전수조사를 해야한다.
  * 버전 관리를 하고 있다면 어떤 버전에서부터 버그가 나타나는지 조사하고 버그가 없는 버전과 비교해서 버그를 찾을 수 있게 됨

* 버전 시간 여행
  * HEAD는 현재 working directory가 어느 버전인지 나타낸다.
  * 버전을 바꾸기 위한 명령(HEAD를 바꾸는 명령) : checkout
    * 현재 버전에서 commit 안 된 것이 있으면 checkout 불가능
    * 작업하다가 잠시 시간여행하려면 브랜치 새로 만들어서 커밋해 둔 후에 과거로 여행갔다오면 될 것 같다.
    * 과거로 돌아가서 파일을 변경 후 커밋한 것은 branch가 없는 상태이기 때문에 다시 checkout하면 사라진다.
  * 시간여행을 끝내고 복귀할 때는 master에서 checkout을 해야한다.

* 실험적인 작업은 브랜치에서 (ex. 쓸지 말지 모르겠는 기능 개발할 때)
  * master 말고 옆 빈 공간에서 우클릭 후 create branch (이름은 exp로 만들어보자)
  * master는 default branch, 나머지는 사용자 정의 branch
  * HEAD를 exp브랜치로 옮기면(checkout) exp브랜치에서 작업할 수 있다.
  * 새로운 버전이 만들어지면 HEAD가 가리키는 브랜치가 그 버전을 따라간다.
  * 버전을 버리는 방법 : delete branch
  * 병합 : merge into current branch
    * A에서 B의 병합과 B에서 A의 병합은 결과물은 동일함
    * exp가 master를 병합 : 업데이트
    * master가 exp를 병합 : 실험이 끝남
      * master로 checkout한다.
      * exp 우클릭 후 merge into current branch

* 로컬 저장소에 원격 저장소를 add
  * Remote - Add Remote
  * Remote name은 그냥 origin으로 하기 (나중에 다른 문제가 생길 수 있음)

* 터미널에서 로그보기
  * `git log --oneline` : git graph 보다 좀 더 분명한 로그를 볼 수 있다.
  * `git log --oneline --graph --all` : 텍스트로 그래프도 그려줌

* 같은 파일을 수정하면 어떻게 되는지
  * pull 안하고 push 하려고 하면 에러가 난다.
    * (복습) pull = fetch + merge
    * fetch를 먼저 한다. (다운로드이기 때문에 conflict가 생기지 않는다.)
    * origin/master에 커서 올려서 merge 하면 된다.
  * 같은 줄을 변경하고 pull을 받으면 충돌이 일어난다.
    * <img src="https://user-images.githubusercontent.com/35680202/129149290-b6f77b84-faf7-4e4b-88d0-9415051d60e2.png" width="600" height="150">
    * 이 힌트를 보고 자기가 직접 병합하면 된다. (vscode가 제공하는 버튼을 누르거나)
    * 충돌이 생겼을 때 공부 먼저 하고 해결하기 : https://opentutorials.org/module/3927
  * 충돌을 막는 방법 : commit하고 이유가 없으면 바로 push 하기

* 팀 작업 시 코드 퀄리티가 중요함 혹은 개선사항 토론
  * pull request (merge request) : (병합을) 요청한다.
  * 기능을 만들 때 브랜치에서 작업하자
    * 브랜치 만들 때 코딩 컨벤션 : ex) feature/login
    * feature/login 브랜치로 checkout 하자
    * 이 브랜치에서 작업한 후 commit & push
  * Compare & pull request : 병합해달라고 하는 요청서 작성하기
    * <img src="https://user-images.githubusercontent.com/35680202/129145235-9f03f8ee-b67b-4517-bd09-588061d0d3c2.png" width="600" height="100">
    * Compare & pull request 버튼을 누른다.
    * <img src="https://user-images.githubusercontent.com/35680202/129145465-51738576-2ec2-45d1-a327-248c3c6c4d50.png" width="600" height="450">
    * 그 와중에 login 2를 또 커밋&푸시하면 pull request에도 올라간다.
    * <img src="https://user-images.githubusercontent.com/35680202/129145654-8152bcce-d2f7-4fc2-b6db-b22d2e39b3aa.png" width="600" height="450">
    * files changed 나 conversation에서 댓글 달아서 의견을 줄 수 있다.
    * <img src="https://user-images.githubusercontent.com/35680202/129146119-80b21001-1363-42e8-be3b-be3cb0cd8afd.png" width="600" height="650">
    * 그 의견을 반영해서 다시 수정한 후 커밋&푸시 (login 3)
    * 이를 반복할 수 있다.
  * master에서 같은 파일을 수정했으면 어떻게 되나요?
    * <img src="https://user-images.githubusercontent.com/35680202/129146507-4b433932-59b0-41f4-b377-06edbcefd41c.png" width="600" height="150">
    * pull request를 달고 commit하면 conflict가 나는지 자동으로 알 수 있다.
    * 계속 작업은 가능한데, 언젠가는 master와 일어난 충돌을 해결하긴 해야함
    * 충돌이 일어날 때마다 미리미리 해결하면 좋긴 하다.
    * feature/login이 자주자주 master를 업데이트해서 충돌이 일어나는지를 확인해야함
    * feature/login에서 master를 fetch&merge 한 후 commit&push 하면 저 위의 상태가 달라진다.
  * Merge pull request 를 누르면 그것에 대한 새로운 commit을 생성하게 되고 commit까지 완성하면 브랜치 merge 완료!
  * pull request 수업자료 : https://opentutorials.org/module/5083

## 오피스 아워

* 선택과제1 - Vision Transformer
  * image imbedding
    * convolution은 이미지 라는 특성을 활용한 것
    * Patch 사이의 관계(attention) 제안
* 선택과제2 - Adversarial autoencoder
  * Autoencoder / VAE / GAN
  * AAE = VAE + GAN
* 선택과제3 - Mixture Density Network
  * 핫한 연구 중 하나
  * 수리통계적인 개념을 많이 알아야 한다.
* 딥러닝 공부
  * 시간을 투자하는 만큼 기억에 잘 남는다. 모르는거는 혼자 좀 찾아보다가 질문하기
  * 부스트캠프 내에 최대한 많이 시도해보고 많이 얻어가라
  * 선택과제는 어려운 것이니 상심하지 말고 차근차근 이해해보자

## 멘토링

* 유튜브 PR12 : 논문 리뷰 영상 참고하면 좋다.
* 논문을 빠르게 파악하는 법 : 일주일에 논문 한 7~10개를 간략하게 요약만 하기 (30분 안에 읽어보기)
* 이미지 분류 대회 준비
  * 시도해보면 좋은 것
    * 모델 & 앙상블 : NFNet, ViT, ResNet, EfficientNet
    * TTA (test time augmentation) : 일종의 앙상블 한 모델로 여러 결과 내는 것
    * 하이퍼파라미터 튜닝, learning rate scheduler 등
    * 실험 기록 잘 해놓기, 중복 실험 관리 등
  * 작년 대회 후기
    * 너무 순위에 집착하면 많은걸 배우지 못할 가능성이 크다.
    * 작년 score 차이가 얼마 안 나서, 순위가 크게 의미있는 것 같지 않다.
  * 꿀팁
    * 대회 때는 주피터 노트북보다 파이썬으로 작업하는 것이 나을 것
    * 파이썬에서 데이터 타입을 typing 하는 습관을 들이자. 여러 명이서 작업할 때 헷갈리지 않아서 좋다.

## 학습 회고

* Transformer 그림만 보고 너무 어렵다고 생각했는데, 시간들여서 계속해서 이해하려고 노력하니까 꽤 많이 정리가 되었다.
* 내일 GAN도 약간 걱정되긴 하지만 열심히 들어야겠다.
