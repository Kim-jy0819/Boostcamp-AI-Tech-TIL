# 부스트캠프 11일차

- [부스트캠프 11일차](#부스트캠프-11일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[1강] Introduction to PyTorch](#1강-introduction-to-pytorch)
    - [[2강] PyTorch Basics](#2강-pytorch-basics)
    - [[3강] PyTorch 프로젝트 구조 이해하기](#3강-pytorch-프로젝트-구조-이해하기)
  - [과제 수행 과정](#과제-수행-과정)
  - [피어 세션 정리](#피어-세션-정리)
  - [특별 강의](#특별-강의)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/17 (화)
  - [x] PyTorch
    - [x] (01강) Introduction to PyTorch
    - [x] (02강) PyTorch Basics
    - [x] (03강) PyTorch 프로젝트 구조 이해하기
    - [x] [필수 과제] Custom Model 개발하기
  - [x] 유석문 CTO님의 특별강의 8/17 (화) 18:00~19:00 - ‘개발자로 산다는 것’

## 강의 내용 정리

### [1강] Introduction to PyTorch

* 프레임워크를 공부하는 것이 곧 딥러닝을 공부하는 것이다.
* 종류
  * PyTorch(facebook)
    * Define by Run (Dynamic Computation Graph) : 실행을 하면서 그래프를 생성하는 방식
    * 개발과정에서 디버깅이 쉽다. (pythonic code)
  * TensorFlow(Google)
    * Define and Run : 그래프를 먼저 정의한 후 실행시점에 데이터를 흘려보냄(feed)
    * Production, Scalability, Cloud, Multi-GPU 에서 장점을 가진다.
    * Keras 는 Wrapper 다. (high-level API)
* 요약
  * PyTorch = Numpy + AutoGrad + Function

### [2강] PyTorch Basics

* Tensor : 다차원 Arrays를 표현하는 클래스 (numpy의 ndarray와 동일)
  * list to tensor, ndarray to tensor 모두 가능
  * tensor는 GPU에 올려서 사용가능
* Operations : numpy 사용법이 거의 그대로 적용된다.
  * reshape() 대신 view() 함수 사용 권장
  * squeeze(), unsqueeze() 차이와 사용법 익히기
  * 행렬곱은 mm(),matmul() 사용 (matmul은 broadcasting 지원)
  * 내적은 dot() 사용
  * `nn.functional` 에서 다양한 수식 변환 지원
* AutoGrad : 자동 미분 지원
  * tensor(requires_grad=True)로 선언한 후 backward() 함수 사용
  * [A GENTLE INTRODUCTION TO `TORCH.AUTOGRAD`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
  * [PYTORCH: TENSORS AND AUTOGRAD](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html)

### [3강] PyTorch 프로젝트 구조 이해하기

* 목표
  * 초기 단계 : 학습과정 확인, 디버깅
  * 배포 및 공유 단계 : 쉬운 재현, 개발 용이성, 유지보수 향상 등
* 방법
  * OOP + 모듈 => 프로젝트
  * 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등을 분리하여 프로젝트 템플릿화
* 템플릿 추천
  * [Pytorch Template](https://github.com/victoresque/pytorch-template) <- 실습 진행📌
  * [Pytorch Template 2](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template)
  * [Pytorch Lightning Template](https://github.com/PyTorchLightning/deep-learning-project-template)
  * [Pytorch Lightning](https://www.pytorchlightning.ai/)
  * [Pytoch Lightning + NNI Boilerplate](https://github.com/davinnovation/pytorch-boilerplate)
* 구글 코랩과 vscode 연결하기
  * colab
    * [ngrok](https://ngrok.com/) 가입하기
    * `colab-ssh` 설치하기
    * 토큰 넣어서 `launch_ssh` 실행해서 연결정보 확인하기
  * vscode
    * extension : Remote - SSH install 하기
    * Remote-SSH: Add New SSH Host 에서 `ssh root@[HostName] -p [Port]` 입력
    * Remote-SSH: Connect to Host 에서 위에서 등록한 host 연결하기
    * `cd /content`로 가면 코랩에서 다운받았던 파일들을 볼 수 있다.
    * `/content/pytorch-template/MNIST-example`에서 `python3 train.py -c config.json` 으로 예제를 실행해볼 수 있다.

## 과제 수행 과정

* Documentaion 살펴보기
  * [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
  * [PyTorch Contribution Guide](https://pytorch.org/docs/stable/community/contribution_guide.html)
  * [`torch.gather`](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather)
    * 임의의 크기의 3D tensor에서 대각선 요소 모으기 : 
      * `indices`를 크기가 `(A.shape[0], min(A.shape[1], A.shape[2]), 1)`인 zero로 채워진 tensor로 선언한다. (`dtype=torch.int64` 임에 주의하기)
      * 이중 for문을 돌면서 `indices[i,j,0] = j` 대각선 요소에 인덱스 값을 넣어준다.
      * `torch.gather(A, 2, indices)` : A의 대각선 요소가 뽑히고, 이를 `view()` 함수로 2D로 만들어주기만 하면 된다.
  * [`nn.Identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html#torch.nn.Identity) 를 사용하는 이유
    * 가독성이나 네트워크를 출력할 때 보이기 위해서
    * if문과 같이 사용하여 코드를 간결하게 만들기 위해서
* Custom 모델 만들기
  * [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
  * `Tensor` vs `Parameter` vs `Buffer`
    * |        항목         | Tensor | Parameter | Buffer |
      | :-----------------: | :----: | :-------: | :----: |
      |    gradient 계산    |   X    |     O     |   X    |
      |     값 업데이트     |   X    |     O     |   X    |
      | 모델 저장시 값 저장 |   X    |     O     |   O    |
    * [`Buffer`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_buffer#torch.nn.Module.register_buffer) : Parameter로 지정하지 않아서 값이 업데이트 되지 않는다 해도
      저장하고싶은 tensor ([예시 : nn.BatchNorm1d 구현 시 사용](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L51-L52))
  * [Docstring](https://en.wikipedia.org/wiki/Docstring) : custom 모델을 만드는 중이고 이 custom 모델을 사용할 다른 개발자들과 미래의 자신을 위해서 Docstring 작성은 필수!
    * [Docstrings in Python - Data Camp](https://www.datacamp.com/community/tutorials/docstrings-python)
  * [hook](https://whatis.techtarget.com/definition/hook) : 프로그래머가 사용자 정의 프로그래밍을 삽입할 수 있도록 하는 패키지 코드로 제공되는 장소이자 인터페이스
    * `self.hooks` 에 등록된 함수가 있으면 실행
    * `self.hooks` 에 등록된 함수가 없으면 무시
    * pre_hook / hook
    * Tensor에 적용하는 (backward) hook / Module에 적용하는 hook
    * `nn.Module`에 등록하는 모든 hook은 `__dict__`를 통해 한번에 확인 가능
  * [apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply) : nn.Module에 이미 구현되어있는 method가 아닌 나만의 custom 함수를 모델에 적용하고 싶을 때 사용하면 좋다.
    * `apply`를 통해 적용하는 함수는 module을 입력으로 받는다.
    * Postorder Traversal 방식으로 module에 함수를 적용한다.
    * 일반적으로 가중치 초기화(Weight Initialization)에 많이 사용된다.
    * elegant way to add a method to an existing object in python
      * ```python
        import types
        class A(object):
            pass
        def instance_func(self):
            print("hi")
        a = A()
        a.instance_func = types.MethodType(instance_func, a)
        ```
  * 참조할 만한 링크
    * 모델 큐레이션 사이트
      * [Browse State-of-the-Art - Papers With Code](https://paperswithcode.com/sota)
      * [labml.ai Annotated PyTorch Paper Implementations - labml.ai](https://nn.labml.ai/)
      * [awesome-deeplearning-resources - endymecy](https://endymecy.github.io/awesome-deeplearning-resources/awesome_projects.html)
    * GitHub 모델 라이센스 체크
      * [Choosing the right license - Github Docs](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/licensing-a-repository#choosing-the-right-license)
      * [Choose an open source license - Choose AI License](https://choosealicense.com/)
      * [오픈소스를 사용하고 준비하는 개발자를 위한 가이드 - if(kakao) dev 2018](https://tv.kakao.com/channel/3150758/cliplink/391717603)

## 피어 세션 정리

- Inception-v4, Inception-ResNet-v1,v2 발표
- 오늘 필수과제에서 Optional 부분을 내일 같이 얘기해보자
- 수업시간에 나온 pytorch-example 코드를 각자 공부해보고 나중에 얘기해보자
- 학습정리 링크공유

## 특별 강의

* 개발놈이 되지 않으려면 어떻게 해야할까
  * 개발자의 필수능력
    * ✨깔끔한 코드✨ : 사람이 이해하기 쉬운 코드, 변경이 용이한 코드, 유지보수 비용 낮은 코드
    * ✨적절한 논리력✨ : 원리 탐색 능력, 제약조건을 고려한 해법, 단순한 디자인
  * 개발시작 전 필수로 구축할 것
    * ATDD(Acceptance Test Driven Development) 
    * ✨**TDD**✨(Test Driven Development)
  * 깔끔한 코드 작성법
    * 사용하는 코드만 만들기(Caller Create) : Dead 코드 지우기, 나중에 필요하면 버전관리에 있는 파일 찾아라
    * 리팩토링(Refactoring) : 위에 드러나는 속성은 그대로 두고 아래에 있는 코드를 변경하는 것, TDD를 작성했으면 이를 쉽게 테스트할 수 있다.
    * 코드 읽기(Code Review) : 나쁜코드는 반면교사, 좋은코드는 배우기
  * 실천법
    * Daily Practice - 몸값 올리는 시간
    * 현재 필요한 만큼만, 간단하게 하라. 미리 더 할 필요없다.
* 좋은 (AI) 개발자?
  * ✨공유✨
    * 장점 : 주변이 똑똑해져야 내가 편하다. 좋은 평판, 주변의 덕 등
    * 방법 : 무엇이듯 공유, 새로운 기술 공유는 내가 직접 써보고 장단점 공유
  * ✨협업✨
    * 전제조건 : 상대를 이해하자
    * 필수요소 : 자아존중감, 내 코드를 지적하는건 나를 지적하는 것이 아님
* 핵심요약
  * 논리력, 좋은 코드 작성 능력
  * 실천력, 피드백
  * 공유, 협업
  * 도메인 지식
* 결론 : 연습이 완벽을 만든다.

## 학습 회고

* 파이토치 나름 써봐서 잘 안다고 생각했는데, 생각보다 더 많은 기능이 있어서 놀랐다. 잘 익혀서 나중에 유용하게 쓰면 좋을 것 같다.
* 파이토치 템플릿도 있는줄 몰랐는데, 앞으로 굉장히 유용하게 사용할 것 같다.
* 좋은 개발자가 되기 위해서 내일 특강도 열심히 들어야겠다.

