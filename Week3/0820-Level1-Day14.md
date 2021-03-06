# 부스트캠프 14일차

- [부스트캠프 14일차](#부스트캠프-14일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[8강] Multi-GPU 학습](#8강-multi-gpu-학습)
    - [[9강] Hyperparameter Tuning](#9강-hyperparameter-tuning)
    - [[10강] Pytorch Troubleshooting](#10강-pytorch-troubleshooting)
    - [[시각화 3강] 차트의 요소](#시각화-3강-차트의-요소)
  - [과제 수행 과정](#과제-수행-과정)
  - [피어 세션 정리](#피어-세션-정리)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/20 (금)
  - [x] PyTorch
    - [x] (08강) Multi-GPU 학습
    - [x] (09강) Hyperparameter Tuning
    - [x] (10강) Pytorch Troubleshooting
    - [x] [선택 과제] Transfer Learning + Parameter Tuning
  - [x] Data Viz : (3강) 차트의 요소
  - [x] 스페셜 피어세션 16:00~17:00
  - [x] 오피스아워 8/20 (금) 18:00 ~ 19:30 Pytorch for AI - 필수과제 해설

## 강의 내용 정리

### [8강] Multi-GPU 학습

* 정의
  * Multi-GPU : GPU를 2개 이상 쓸 때
  * Single Node Multi GPU : 한 대의 컴퓨터에 여러 개의 GPU
* 방법
  * 모델 병렬화(Model parallel)
    * 모델을 나누기 (ex. AlexNet)
    * 모델의 병목, 파이프라인 어려움 등의 문제
    * ```python
      # __init__()
      self.seq1 = nn.Sequential(~~).to('cuda:0') # 첫번째 모델을 cuda 0에 할당
      self.seq2 = nn.Sequential(~~).to('cuda:1') # 두번째 모델을 cuda 1에 할당
      # forward()
      x = self.seq2(self.seq1(x).to('cuda:1')) # 두 모델 연결하기
      ```
  * 데이터 병렬화(Data parallel)
    * 데이터를 나눠 GPU에 할당한 후, 결과의 평균을 취하는 방법
    * `DataParallel`
      * 특징 : 단순히 데이터를 분배한 후 평균 (중앙 코디네이터 필요)
      * 문제 : GPU 사용 불균형, Batch 사이즈 감소, GIL(Global Interpreter Lock)
      * `parallel_model = torch.nn.DataParallel(model)`
    * `DistributedDataParallel` : 각 CPU마다 process 생성하여 개별 GPU에 할당
      * 특징 : 개별적으로 연산의 평균을 냄 (중앙 코디네이터 불필요, 각각이 코디네이터 역할 수행)
      * 방법 : 각 CPU마다 process 생성하여 개별 GPU에 할당 (CPU도 GPU 개수만큼 할당)
      * ```python
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, 
            shuffle=False, pin_memory=True, num_workers=[GPU개수x4],
            sampler=train_sampler # sampler 사용
        )
        ...
        # Distributed dataparallel 정의
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        ```
      * 더 자세한 코드는 [여기](https://blog.si-analytics.ai/12)
* 추가자료
  * [PyTorch Lightning - MULTI-GPU TRAINING](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html)
  * [PyTorch - GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  * [관련 집현전 영상](https://youtu.be/w4a-ARCEiqU?t=1978)
  * TensorRT : NVIDIA가 제공하는 도구

### [9강] Hyperparameter Tuning

* 목표 : 마지막 0.01의 성능이라도 높여야 할 때!
* 기법 : Grid Search / Random Search / 베이지안 기법(BOHB 등)
* ✨[Ray](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)✨ : multi-node multi processing 지원 모듈, Hyperparameter Search를 위한 다양한 모듈 제공
  * 방법
    * `from ray import tune`
    * config에 search space 지정
      * ```python
        config = {
                "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": tune.choice([2, 4, 8, 16])}
        ```
    * 학습 스케줄링 알고리즘 지정
      * ```python
        scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2)
        ```
    * 결과 출력 양식 지정
      * ```python
        reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=["loss", "accuracy", "training_iteration"])
        ```
    * 병렬 처리 양식으로 학습 실행
      * ```python
        result = tune.run(
                partial(train_cifar, data_dir=data_dir), # train_cifar : full training function
                resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                progress_reporter=reporter)
        ```
    * 학습 결과 가져오기
      * `best_trial = result.get_best_trial("loss", "min", "last")`

### [10강] Pytorch Troubleshooting

* OOM(Out Of Memory) : 왜, 어디서 발생했는지 알기 어려움
* 쉬운 방법 : Batch size 줄이고 GPU clean 한 후 다시 실행
* 다른 방법
  * `GPUtil.showUtilization()` : GPU의 상태를 보여준다.
  * `torch.cuda.empty_cache()` : 사용되지 않은 GPU상 cache 정리 (학습 loop 전에 실행하면 좋다)
  * `total_loss += loss.item` : `total_loss += loss` 에서는 계산 그래프를 그릴 필요가 없기 때문에,  python 기본 객체로 변환하여 더해준다.
  * `del 변수` : 필요가 없어진 변수는 적절히 삭제하기 (정확히는 변수와 메모리 관계 끊기)
  * try-except문을 이용해서 가능한 batch size 실험해보기
  * `with torch.no_grad()` : Inference 시점에 사용
  * tensor의 float precision을 16bit로 줄일 수도 있다. (많이 쓰이진 않음)
* [이 외 GPU 에러 정리 블로그](https://brstar96.github.io/shoveling/device_error_summary/)

### [시각화 3강] 차트의 요소

* Text 사용하기
  * 시각적으로만 표현이 불가능한 내용 설명
* Color 사용하기
  * 효과적인 구분, 색조합 중요, 인사이트 전달 중요
  * 범주형 : 독립된 색상 / 연속형 : 단일 색조의 연속적인 색상 / 발산형 : 상반된 색
  * 색각 이상(색맹, 색약) 고려가 필요할 수 있다.
* Facet(분할) 사용하기
  * Multiple View : 다양한 관점 전달
  * subplot, grid spec 등 이용

## 과제 수행 과정

* [Xavier Initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)

## 피어 세션 정리

* 다음 주 목표 : 최선을 다 하고, 많이 배우는 시간이 될 수 있도록 하자!
* EfficientNet-v2
  * small image size를 사용하는 FixRes 기법 사용
  * Fused-MBConv : 모델의 앞부분의 Depthwise Convolution의 training speed가 느린 문제가 있어 앞부분의 MBConv를 Fused-MBConv(1X1 Conv와 DWConv를 3X3 Conv로 대체)를 사용
  * Progressive learning : 큰 image size에 대한 낮은 정확도를 보완하기 위해 [RandAug](https://arxiv.org/abs/1909.13719)를 image size에 따라 강도를 다르게 적용시킴

## 학습 회고

* 부스트캠프 생활이 생각보다 많이 익숙해져서, 이제 MLOps 스터디랑 알고리즘 스터디도 병행하게 되었다.
* 제대로 안 할거면 들어가지도 않았을거니까 계획한만큼 열심히 공부해봐야겠다.

