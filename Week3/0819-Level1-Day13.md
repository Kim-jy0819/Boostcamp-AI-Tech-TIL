# 부스트캠프 13일차

- [부스트캠프 13일차](#부스트캠프-13일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[6강] 모델 불러오기](#6강-모델-불러오기)
    - [[7강] Monitoring tools for PyTorch](#7강-monitoring-tools-for-pytorch)
    - [[시각화 2강] 기본 차트의 사용](#시각화-2강-기본-차트의-사용)
  - [피어 세션 정리](#피어-세션-정리)
  - [마스터 클래스](#마스터-클래스)
  - [멘토링](#멘토링)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/19 (목)
  - [x] PyTorch
    - [x] (06강) 모델 불러오기
    - [x] (07강) Monitoring tools for PyTorch
  - [x] Data Viz : (2강) 기본 차트의 사용
  - [x] 마스터 클래스 8/19 (목) 18:00 ~ 19:00 최성철 마스터님 - Data centric AI
  - [x] 멘토링 20:00~

## 강의 내용 정리

### [6강] 모델 불러오기

* 모델의 파라미터 저장 및 로드
  * `model.state_dict()` : 모델의 파라미터 표시
  * ```python
    torch.save(model.state_dict(),"model.pt") # 저장
    new_model = ModelClass()
    new_model.load_state_dict(torch.load("model.pt")) # 로드
    ```
* 모델 형태(architecture)와 파라미터 저장 및 로드
  * ```python
    torch.save(model, "model.pt") # 저장
    new_model = torch.load("model.pt") # 로드
    ```
* 모델 구조 출력
  * ```python
    from torchsummary import summary
    summary(model, (3, 224, 224))
    ```
  * ```python
    for name layer in model.named_modules():
        print(name, layer)
    ```
* checkpoints
  * 학습의 중간 결과 저장, 일반적으로 epoch, loss, mertic을 함께 저장
  * ```python
    torch.save({
        'epoch':e, 'loss': epoch_loss, 'optimizer_state_dict': optimizer.state_dict(),
        'model_statae_dict': model.state_dict()
    }, PATH) # 저장
    checkpoint = torch.load(PATH) # 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    ```
* pretrained model transfer learning
  * 다른 데이터셋(일반적으로 대용량 데이터셋)으로 만든 모델을 현재 데이터에 적용
    * 모델 추천 : [CV models](https://github.com/rwightman/pytorch-image-models) & [NLP models](https://huggingface.co/models)
  * 마지막 레이어 수정하기
    * `vgg.fc = torch.nn.Linear(1000, 1)` : 맨 마지막에 fc 레이어 추가하기 (추천)
    * `vgg.classifier._modules['6'] = torch.nn.Linear(4096, 1)` : 맨 마지막 레이어 교체하기
  * Freezing : pretrained model 활용 시 모델의 일부분을 frozen 시킨다.
    * ```python
      for param in mymodel.parameters():
          param.requires_grad = False # frozen
      for param in mymodel.linear_layers.parameters():
          param.requires_grad = True # 마지막 레이어 살리기
      ```

### [7강] Monitoring tools for PyTorch

* 목표 : print문은 이제 그만!
* ✨[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)✨ : TensorFlow의 프로젝트로 만들어진 시각화 도구, PyTorch도 연결 가능
  * 종류
    * scalar : metric 등 표시
    * graph : 계산 그래프 표시
    * histogram : 가중치 등 값의 분포 표시
    * image / text : 예측값과 실제값 비교
    * mesh : 3d 형태로 데이터 표형 (위에 비해 자주 쓰진 않음)
  * 방법
    * 기록을 위한 디렉토리 생성 : `logs/[실험폴더]` 로 하는 것이 일반적
    * 기록 생성 객체 `SummaryWriter` 생성
    * `writer.add_scalar()` 등으로 값들을 기록
    * `writer.flush()` : disk에 쓰기
    * `%load_ext tensorboard` : 텐서보드 부르기
    * `%tensorboard --logdir {logs_base_dir}` : 6006포트에 자동으로 텐서보드 생성
* ✨[Weight & Biases(WandB)](https://wandb.ai/site)✨ : 머신러닝 실험 지원, MLOps의 대표적인 툴이 되고 있다.
  * 기능
    * 협업 / code versioning / 실험 결과 기록 등
    * 유료지만, 무료 기능도 있다.
  * 방법
    * 홈페이지 : 회원가입 -> API 키 확인 -> 새 프로젝트 생성
    * `wandb.init(project, entity)` : 여기서 API 입력해서 접속
    * `wandb.init(project, config)` : config 설정
    * `wandb.log()` : 기록
* [Pytorch Lightning Logger 목록](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)

### [시각화 2강] 기본 차트의 사용

* Bar plot : 막대 그래프, 범주에 따른 수치값 비교에 적절
  * Principle of Proportion ink : 실제 값과 그래픽으로 표현되는 잉크 양은 비례해야 함
* Line plot : 꺾은선 그래프, 시간/순서에 대한 추세(시계열 데이터)에 적합
  * 보간 : 점과 점 사이에 데이터가 없을 때 잇는 방법
* Scatter plot : 산점도 그래프, 두 feature간의 상관 관계 파악에 용이
  * 인과 관계(causal relation)과 상관 관계(correlation)은 다르다.

## 피어 세션 정리

* EfficientNet
  * Neural Architecture Search를 사용해서 baseline 네트워크를 만듦
  * depth, width, resolution을 동시에 scaling : 상수배를 이용한다.
  * 기존 모델들보다 더 효율적이고 성능이 좋다.
  * b0~b7 모델이 있는데, 한 단계 올라갈 때마다 크기가 두 배가 된다.

## 마스터 클래스

* 요즘 경향
  * Pretrained-Learning
  * 성능을 올리기 위한 모델 싸움은 옛날만큼 치열하지 않다.
  * 모델이 아니라 데이터를 중심으로 AI 시스템을 개발하고자 한다.
* Research ML
  * 데이터는 이미 준비됨 -> 열심히 모델 개발 -> 성능 확인 -> 논문 작성
  * 즉, 모델 개발 / 하이퍼파라미터 튜닝 싸움
* Project-Real World ML
  * 문제를 먼저 정의한 후 데이터 확보, 모델 개발..., 최종적인 서비스
  * 현실에서 ML code 는 매우 작은 부분
* 데이터 : 양질의 데이터 확보가 관건
  * Data drift : 시간이 지나면서 데이터는 계속 바뀐다.
  * 기업 비교
    * **테슬라** : 유저가 데이터를 만들어준다.
    * 웨이모 : 시뮬레이션으로 데이터를 만든다.
  * Data Feedback Loop
    * 사용자로부터 오는 데이터를 자동화하여 모델에 피딩해주는 체계가 필요하다.
    * 대용량 데이터 다루기, 자동화하는 역량을 가지는 것이 중요
* 앞으로 알아야할 것들
  * MLOPs 도구들
  * 데이터베이스, **SQL** : 완전 필수
  * Clooud 서비스 - AWS, GCP, Azure : 회사에서 당연히 쓰인다.
  * **Spark** (+Hadoop) : 깊지 않더라도 기본적인 코드는 읽을 줄 알아야 한다.
  * **Linux** + Docker + 쿠버네티스 : 필수
  * 스케줄링 도구들 - 쿠브플로우, MLFlow, **AirFlow** : 한번쯤은 써봐야 한다.
* 🔥핵심 요약🔥
  * 앞으로는 알고리즘 연구자 보다 **ML/DL 엔지니어의 필요성**이 더 증대
  * 단순히 ML/DL 코드 작성을 넘어서야 함 → 자동화하고, 데이터와 연계, 실험 결과를 기반으로 설득, 시스템화
  * 좋은 엔지니어이자 **좋은 기획자적인 요소**들이 필요 → 아직 AI화 되지 않는 영역의 AI화 (데이터를 어떻게 먹일 것인가!)
  * Shell script, 네트워크, 클라우드 등 기본적으로 알아야 할 것들이 많음

## 멘토링

* 커스텀 데이터셋 만드는 것 익숙해지면 좋다. (Multi-GPU)
* Auto ML : 레이, **옵튜나** 등 사용해보면 좋다.
* MLOps는 수요가 높아질 것이다. 근데 당장은 MLOps를 도입한 회사는 거의 없다.
* 당장 MLOps를 공부하기보다는 지금 공부하고 있는걸 잘 정리하는게 좋을 것 같다.
* 깃헙에 공부한 내용이나 도움이 될만한 코드를 정리해서 다른 사람들에게 공유하면 좋을 것 같다.
* 부스트캠프 목표 : 취업을 목표로 하지는 않았으면 좋겠다. 관심분야를 탐색하는 시간, 적당히 컨디션 조절도 잘 하기
* Attention is all you need 설명 영상 리스트
  1. https://www.youtube.com/watch?v=iDulhoQ2pro
  2. https://www.youtube.com/watch?v=4Bdc55j80l8
  3. https://www.youtube.com/watch?v=U0s0f995w14

## 학습 회고

* 오늘은 체력적으로 정말 힘들었지만, 나름대로 목표를 다시 세워보면서 다시 기운을 차려보았다.
* 이번 주까지는 부캠 스케줄 적응도 하고 공부도 열심히 했으니까 이제 하고 싶은 것도 다 하려고 한다!

