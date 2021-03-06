# 부스트캠프 8일차

- [부스트캠프 8일차](#부스트캠프-8일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[4강] Convolution](#4강-convolution)
    - [[5강] Modern CNN](#5강-modern-cnn)
    - [[6강] Computer Vision Applications](#6강-computer-vision-applications)
  - [과제 수행 과정](#과제-수행-과정)
  - [특강 내용 정리](#특강-내용-정리)
  - [피어 세션 정리](#피어-세션-정리)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/11 (수)
  - [x] DL Basic
    - [x] (4강) CNN - Convolution은 무엇인가?
    - [x] (5강) Modern CNN - 1x1 convolution의 중요성
    - [x] (6강) Computer vision applications
    - [x] [필수 과제3] CNN Assignment
  - [x] 이고잉님의 Git/Github 특강 13:00~15:30

## 강의 내용 정리

### [4강] Convolution

* 2D Image Convolution
  * <!-- $(I * K)(i, j) = \sum_m \sum_n I(m,n) K(i-m, j-n) = \sum_m \sum_n I(i-m, i-n) K(m,n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=(I%20*%20K)(i%2C%20j)%20%3D%20%5Csum_m%20%5Csum_n%20I(m%2Cn)%20K(i-m%2C%20j-n)%20%3D%20%5Csum_m%20%5Csum_n%20I(i-m%2C%20i-n)%20K(m%2Cn)">
* RGB Image Convolution
  * ![image-20210811103921000](https://user-images.githubusercontent.com/35680202/128956799-093d9612-2e44-455e-bbe3-7bfb986e7d70.png)
* Stack of Convolutions
  * Conv -> ReLU -> Conv -> ReLU
* Convolutional Neural Networks
  * feature extraction : conv layer, pooling layer
  * decision making : fully connected layer
* 1x1 Convolution
  * 목표 : Dimension(채널) reduction - 파라미터 수 줄이기
  * ex) bottleneck architecture

### [5강] Modern CNN

* ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)
  * 분류(Classification) : 1000개의 카테고리
  * 데이터셋 : 100만 장 이상, 학습 데이터셋 : 45만 장
  * Human 성능 : 5.1% <- 2015년도부터 사람의 성능을 따라잡기 시작
* **AlexNet**
  * 구조
    * 5 conv layers + 3 dense layers = 8 layers
    * 11x11 filters : 파라미터가 많이 필요하다.
  * 핵심
    * ReLU(Rectified Linear Unit) activation 사용 : vanishing gradient problem 해결
    * 2개의 GPU 사용
    * Data augmentation, Dropout
  * 의의 : 일반적으로 제일 잘 되는 기준을 잡아준 모델
* **VGGNet**
  * 구조
    * **3x3 filters 만 사용**
    * 1x1 conv 를 fully connected layers 에서 사용
    * 레이어 개수에 따라서 VGG16, VGG19
    * Dropout (p=0.5)
  * 핵심
    * 3x3 conv 두 번이면, 5x5 conv 한 번과 receptive field가 (5x5)로 같다.
      * 3x3 conv 두 개의 파라미터 수 : (3x3x128x128) x2 = 294,912
      * 5x5 conv 한 개의 파라미터 수 : (5x5x128x128) x1 = 409,600
* **GoogLeNet**
  * 구조
    * 22 layers
    * network in network : 비슷한 네트워크가 네트워크 안에서 반복됨
    * Inception blocks
      * ![image](https://user-images.githubusercontent.com/35680202/128962913-0ce88766-ceb8-4511-ab3a-d8063d161620.png)
      * 여러 개의 receptive field를 가지는 filter를 거치고, 결과를 concat 하는 효과
      * 1x1 conv 가 끼워지면서 전체적인 파라미터 수를 줄일 수 있게 된다. (channel-wise dimension reduction)
  * 핵심
    * **1x1 conv 의 이점**
      * (1) in_c=128, out_c=128, **3x3 conv** 의 파라미터 수 : (3x3x128x128) = 147,456
      * (2-1) in_c=128, out_c=32, **1x1 conv** 의 파라미터 수 : (1x1x128x32) = 4096
      * (2-2) in_c=32, out_c=128, **3x3 conv** 의 파라미터 수 : (3x3x32x128) = 36,864
      * (1) >>> (2-1)+(2-2)
* **ResNet**
  * 문제 : 깊은 네트워크가 학습시키기 어렵다. 오버피팅 아니고, 학습이 잘 안 되는 것
  * 구조
    * Identity map (**skip connection**, residual connection)
      * ![image](https://user-images.githubusercontent.com/35680202/128962878-0a05ee59-c227-4eb4-837b-8a2393228ab0.png)
    * Shortcut
      * Simple Shortcut : 입력 x 그대로 사용(차원이 같은 경우)
      * Projected Shortcut : x 에 1x1 conv를 통과시켜 channel depth를 match 시킨다.
    * Bottleneck architecture
      * ![image](https://user-images.githubusercontent.com/35680202/128963667-aecf7b9c-f8d4-449c-bd21-6c3d29c37b39.png)
      * 3x3 conv 하기 전/후에 1x1 conv 로 채널 수를 줄이고, 다시 늘리는 방법
  * 의의
    * 네트워크를 더 깊게 쌓을 수 있는 가능성을 열어줌
    * Performance는 증가하는데 parameter size는 감소하는 방향으로 발전
* **DenseNet**
  * 구조
    * Dense Block
      * ![image](https://user-images.githubusercontent.com/35680202/128963758-d6d7ea6e-0651-493a-aaf7-39940e3706f6.png)
      * addition 대신 **concatnation** 을 사용한다.
      * 채널이 기하급수적으로 커지게 된다.
    * Transition Block (for Dimension reduction)
      * batchnorm -> 1x1 conv -> 2x2 avgpool
  * 웬만하면 resnet이나 densenet을 쓰면 성능이 좋다.

### [6강] Computer Vision Applications

* **Semantic Segmentation**
  * 이미지의 모든 픽셀이 어떤 라벨에 속하는지 보고 싶은 것
  * 자율주행에 가장 많이 활용됨
  * **fully convolutional network**
    * convolutionalization
      * dense layer를 없애고 싶음
      * 파라미터의 수는 똑같다.
      * ![image](https://user-images.githubusercontent.com/35680202/128982583-c7d609f2-4b7d-465b-b058-147f4d07f547.png)
    * 특징
      * Input 의 spatial dimension 에 독립적이다.
      * heatmap 같은 효과가 있다.
    * Deconvolution (conv transpose)
      * dimension이 줄어들기 때문에 upsample 이 필요하다.
      * convolution의 역 연산이라고 생각하면 편함
      * spatial dimension을 키워준다.
* **Detection**
  * 어느 객체가 어디에 있는지 bounding box 를 찾고 싶은 것
  * R-CNN
    * 방법
      * 이미지 안에서 region을 여러 개 뽑는다. (Selective search)
      * 똑같은 크기로 맞춘다.
      * 피쳐를 뽑아낸다. (AlexNet)
      * 분류를 진행한다. (SVM)
    * 문제
      * region을 뽑은 만큼 CNN에 넣어서 계산해야하니까 계산량도 많고 오래 걸린다.
  * SPPNet
    * 목표 : 이미지를 CNN에서 한번만 돌리자
    * 방법
      * 이미지 안에서 bounding box를 뽑고, 
      * 이미지 전체에 대해서 convolutional feature map을 만든 다음에, 
      * 뽑힌 bounding box에 해당하는 convolutional feature map의 텐서만 가져와서 쓰자
      * 결론적으로, CNN을 한번만 돌아도 텐서를 뜯어오는 것만 region별로 하기 때문에 훨씬 빨라진다.
    * 한계
      * 여전히 여러 개의 region을 가져와서 분류하는 작업이 필요
  * Fast R-CNN
    * 방법
      * SPPNet과 거의 동일한 컨셉
      * 뒷단에 neural network와 RoI feature vector를 통해서 bounding box regression과 classification을 했다는 점
  * Faster R-CNN
    * 목표 : bounding box를 뽑는 것도 network로 학습하자
    * 방법 : **Region Proposal Network(RPN)** + Fast R-CNN
      * Region Proposal Network(RPN) : 이미지의 특정 영역(패치)가 bounding box로서의 의미가 있을지 없을지를 찾아준다. 물체가 무엇인지는 뒷단에 있는 네트워크가 해줌
        * Anchor boxes : 미리 정해놓은 bounding box의 크기 (대충 이 이미지에 어떤 크기의 물체가 있을 것 같은지 정함, 템플릿같은 것)
        * ![image](https://user-images.githubusercontent.com/35680202/129042366-6e6a755c-476b-492f-bfa8-dd962f90dd58.png)
  * YOLO (v1)
    * 목표 : No explicit bounding box sampling
      * You Only Look Once
      * Faster R-CNN 보다 훨씬 빠르다.
    * 방법
      * 이미지가 들어오면 SxS grid 로 나눈다.
      * 찾고 싶은 물체의 중앙이 해당 grid 안에 들어가면 그 grid cell이 해당 물체에 대한 bounding box와 그 해당 물체가 무엇인지를 같이 예측
      * 각각의 cell은 B개의 bounding box 예측 + C개의 class에 대한 probabilities
    * 정리
      * SxS x (B*5 + C) tensor
        * SxS : Number of cells of the grid
        * B*5 : B bounding boxes with offsets (x,y,w,h) and confidence(필요성)
        * C : Number of classes

## 과제 수행 과정

* 필수과제3 - CNN
  * `stride=1`일 때, `padding=(kernel_size//2)` 이면 `in_H == out_H` 되게 만들 수 있다.
  * `nn.Sequential()`로 선언하고, `add_module(layer_name, layer)`로 레이어를 쌓으면 이름을 지정해줄 수 있어서 좋은 점이 있다.

## 특강 내용 정리

* [이고잉님 특강 모아 보기](https://seomal.com/map/1)
* 서론
  * 개발자 도구 어떻게 하면 일반인들도 잘 사용할 수 있을까
  * 리눅스(의 버전관리)를 위해서 시작한 프로젝트가 깃
  * 깃허브는 개발자의 SNS 같은 것
* Commit
  * 끝난 하나의 단위 작업을 커밋
  * 커밋 기록에서 browse files 를 누르면 그 commit을 할때의 컴퓨터 상태를 볼 수 있음
* Issue : 저장소에 있는 게시판, 업무에 대한 협의를 하기 위한 최적의 게시판
  * new issue
    * Labels : 이슈의 성격을 나타냄
    * Assignees : 동료를 지정하면 그 사람에게 이메일이 간다.
  * Close Issue 하면 이 작업이 끝났다는 것을 알려줄 수 있다.
  * 파일에서 줄 선택 후 Reference new issue할 수도 있다.
* Wiki : 프로젝트 메뉴얼 등을 위키로 정리
* vscode
  * new window -> clone repository 해서 원격 저장소를 받아올 수 있다.
  * ctrl + , : setting page
  * Extensions - Git Graph 설치
  * Source Control - 파일 수정 기록을 볼 수 있다.
* 분산 버전 관리
  * main : 로컬 저장소에 저장된 것
  * origin/main : 원격 저장소에 저장된 것
  * push : 내 컴퓨터에서 원격 저장소로 올리는 것
* 깃의 장점
  * 언제 동기화할지 내가 정할 수 있다.
  * 어떤 부분이 변경되었는지 볼 수 있다.
* 여러 사람이 동시에 커밋 : 나중에 푸시하려는 사람이 pull을 땡겨와서 merge된 걸 push 해야 함
  * pull = fetch + merge
  * fetch : 다운로드
  * merge : 합치기

## 피어 세션 정리

* GoogLeNet
  * inception module : concat 방법 - padding으로 차원을 맞춰주고 channel side에서 합친다.
  * auxiliary classifier
  * global average pooling
* ResNet
  * Residual block의 backpropagation : 뒤에서 전달되는 gradient가 그대로 x에도 전달되기 때문에 vanishing gradient 문제를 완화할 수 있다.

## 학습 회고

* 깃을 꽤 쓸줄 안다고 생각했는데, 특강을 통해 여러 꿀팁과 용어에 대한 설명을 들을 수 있어서 좋았다.
* 논문을 구현해보는게 확실히 도움이 될 것 같아서 주말에 시도해볼까 생각 중이다.
