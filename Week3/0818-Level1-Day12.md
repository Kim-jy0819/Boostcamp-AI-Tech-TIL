# 부스트캠프 12일차

- [부스트캠프 12일차](#부스트캠프-12일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 내용 정리](#강의-내용-정리)
    - [[4강] AutoGrad & Optimizer](#4강-autograd--optimizer)
    - [[5강] Dataset & Dataloader](#5강-dataset--dataloader)
    - [[시각화 1강] Introduction to Visualization (OT)](#시각화-1강-introduction-to-visualization-ot)
  - [과제 수행 과정](#과제-수행-과정)
  - [피어 세션 정리](#피어-세션-정리)
  - [특별 강의](#특별-강의)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/18 (수)
  - [x] PyTorch
    - [x] (04강) AutoGrad & Optimizer
    - [x] (05강) Dataset & Dataloader
    - [x] [필수 과제] Custom Dataset
  - [x] Data Viz : (1강) Introduction to Visualization
  - [x] 유석문 CTO님의 특별강의 8/18 (수) 18:00~19:00 - ‘파이썬 Unit Test’

## 강의 내용 정리

### [4강] AutoGrad & Optimizer

* 딥러닝 아키텍쳐 : 블록 반복의 연속
  * Layer = Block
* `nn.Module` : Layer의 base class
  * Input, Output, Forward, Backward, parameter 정의
* `nn.Parameter` : Tensor 객체의 상속 객체
  * `nn.Module` 내에서 parameter로 정의될 때, `required_grad=True` 지정하기
  * `layer.parameter()`에는 `required_grad=True` 로 지정된 변수들만 포함된다.
  * 대부분의 layer에 weights 값들이 지정되어 있어서 직접 지정할 일은 거의 없긴 함
* Backward from the scratch
  * `nn.Module`에서 `backward`와 `optimizer` 오버라이딩하면 된다.
* 추가자료
  * [Pytorch로 Linear Regression하기](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817)
  * [Pytorch로 Logistic Regession하기](https://medium.com/dair-ai/implementing-a-logistic-regression-model-from-scratch-with-pytorch-24ea062cd856)

### [5강] Dataset & Dataloader

* `Dataset(Data, transforms)` : 데이터를 입력하는 방식의 표준화
  * `__init__()` : 초기 데이터 생성 방법 지정
  * `__len__()` : 데이터의 전체 길이
  * `__getitem__()` : index값을 주었을 때 반환되는 데이터의 형태
* `DataLoader(Dataset, batch, shuffle, ...)` : 데이터의 batch를 생성해주는 클래스
* [TORCHVISION.DATASETS](https://pytorch.org/vision/stable/datasets.html)
  * [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) : 위의 소스코드 참고해서 데이터셋 만드는 연습해보기
* 추가자료
  * [Pytorch Dataset, Dataloader 튜토리얼](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### [시각화 1강] Introduction to Visualization (OT)

* 시각화
  * 구성요소 : 목적, 독자, 데이터, 스토리, 방법, 디자인
  * 목표 : 모범 사례를 통해 좋은 시각화를 만들어보자
* `데이터` 시각화
  * 데이터셋 종류 : 정형, 시계열, 지리, 관계형, 계층적, 비정형 데이터
  * 수치형(numerical) : 연속형(continuous) / 이산형(discrete)
  * 범주형(categorical) : 명목형(nominal) / 순서형(ordinal)
* 시각화의 요소
  * 마크(mark) : 점, 선, 면
  * 채널(channel) : 마크를 변경할 수 있는 요소들
    * 위치, 색, 모양, 크기, 부피, 각도 등
    * 전주의적 속성(Pre-attentive Attribute)
      * 주의를 주지 않아도 인지하게 되는 요소
      * 적절하게 사용할 때, 시각적 분리(visual pop-out)
* `Matplotlib` : `numpy`와 `scipy`를 베이스로 하여, 다양한 라이브러리와 호환성이 좋다.

## 과제 수행 과정

* Dataset 관련 모듈
  * [`torch.utils.data`](https://pytorch.org/docs/stable/data.html) : 데이터셋의 표준을 정의하고 데이터셋을 불러오고 자르고 섞는데 쓰는 도구들이 들어있는 모듈
    * `torch.utils.data.Dataset` : 파이토치 모델을 학습시키기 위한 데이터셋의 표준 정의
    * `torch.utils.data.DataLoader` : Dataset 모듈을 상속하는 파생 클래스를 입력으로 사용
  * [`torchvision.dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) : `torch.utils.data.Dataset`을 상속하는 이미지 데이터셋의 모음 (ex) MNIST, CIFAR-10
    * `random_split`을 이용해서 Train / Validatoin 데이터셋도 손쉽게 분할 가능
    * `datasets.ImageFolder`를 통해 폴더를 아래와 같이 클래스별로 나누고 해당 경로를 지정해두면 손쉽게 Dataset을 생성할 수 있다.
  * [`torchtext.dataset`](https://pytorch.org/text/stable/datasets.html) : `torch.utils.data.Dataset`을 상속하는 텍스트 데이터셋의 모음 (ex) IMDb, AG_NEWS
  * [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) : 이미지 데이터셋에 쓸 수 있는 여러 가지 변환 필터를 담고 있는 모듈
    * 주의 : torchvision은 항상 PIL 객체로 받거나 `ToPILImage()` 사용하기
    * `Compose`를 이용하면 여러 transforms를 한꺼번에 처리할 수 있다.
    * [albumentations](https://github.com/albumentations-team/albumentations), [imgaug](https://github.com/aleju/imgaug)와 같은 라이브러리도 있다.
  * [`torchvision.utils`](https://pytorch.org/vision/stable/utils.html) : 이미지 데이터를 저장하고 시각화하기 위한 도구가 들어있는 모듈
* torch.Tensor 객체를 만들 때, `df.values`를 통해 dataframe 객체를 numpy array 객체로 바꾼다.
* DataLoader의 `collate_fn` 파라미터
  * ((피처1, 라벨1) (피처2, 라벨2))와 같은 배치 단위 데이터가 ((피처1, 피처2), (라벨1, 라벨2))와 같이 바꿀 수 있다.
  * 같은 batch에서 길이가 다른 sample들의 길이를 일정하게 맞춰줄 수 있다. ([`nn.functional.pad()` 이용하여 패딩](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html))
* vocab을 만들 때, lower() 해서 만들기 때문에, 어떤 문자열에 대한 encode 값을 얻으려면 전처리를 해야한다.
  * 특수문자 제거 : `s = re.sub('[\W]', ' ', s)`
  * 두 개 이상의 공백 제거 : `s = ' '.join(s.split())`
  * 소문자로 만들기 : `s = s.lower()`
  * 앞 뒤 공백 지우기 : `s = s.strip()`
* [HuggingFace Dataset](https://huggingface.co/docs/datasets/)를 이용하면 훨씬 편하게 Dataset와 DataLoader를 사용할 수 있다.

## 피어 세션 정리

* ResNeXt
  * Cardinality 기법을 이용해서 ResNet의 성능을 높이려고 했다.
  * Deep(Layer) 보다 Width, Width 보다 Cardinality를 증가시키는 것이 더 효율적이다.
  * 동일한 파라미터 수 대비 더 좋은 성능을 낸다.

## 특별 강의

* 책 추천 : refactoring to patterns
* 테스트를 하면 좋은 이유
  * 왜 잘되는지, 왜 안되는지 더 잘 분석/이해할 수 있다.
  * 기획문서는 보통 업데이트가 잘 안되서 그걸 기반으로 테스트를 하기가 어렵다.
  * 디버거를 잘 쓰는 것보다 오류가 있을 때 바로 발견할 수 있는 코드를 짜는 것이 더 중요하다.
  * Regression failure : 버그 하나를 고쳤는데 다른데서 새로운 버그가 생길 수 있다. 자동화된 케이스가 없다면 발견하기 어렵다.
  * 버그가 늦게 발견될수록 비용이 훨씬 더 많이 든다.
* 테스트 방법
  * ATDD : Customer's Tests
  * TDD : Programmer's Tests
    * Unit Tests : dependency 없이 테스트, 빠르게 내가 작성한 부분을 고칠 수 있다.
    * Intergration Tests : dependency까지 고려해서 정상동작 하는지 확인, 어디가 잘못인지 식별을 해야한다.
  * ✨Unit Test✨가 가장 기반이 된다. 이게 잘 되면 그 다음 단계를 진행할 수 있다.
* Maturity model
  * Manual testing : (일반적인 방법) 손으로 테스트, 시간낭비가 심하다.
  * Automated testing : 자동화 테스트, 코드를 먼저 만들고 테스트 코드를 만들려고 하면 어렵다.
  * Test-first development : 설계를 한 후에 테스트를 작성하고 테스트를 통과하도록 코드 작성하기
  * ✨Test-driven developmnet✨ : 테스트를 작성하면서 설계도 같이 정의한 후에 코드를 작성하기
* 🚀연습하기 : Unit Test 작성하기 -> 테스트를 통과할만한 코드 짜기 -> 리팩토링 하기🚀
* [Python Unit Test](https://docs.python.org/3/library/unittest.html)
  * ```python
    import unittest
    
    def add(x, y):
        return x+y
    
    class SimpleTest(unittest.TestCase):
    	def testadd(self):
    		self.assertEqual(add(4,5), 9)
            
        def testaddfail(self):
            self.assertEqual(add(-0.5,1),1.5)
    
    if __name__ == '__main__':
    	unittest.main()
    ```
  * 요소
    * test fixture : 테스트 하는데 도움이 되는 것들을 준비하는 과정
    * test case : 실제 테스트할 것들
    * test suite : 테스트를 목적(개발 단계, 커밋 단계 등)에 따라서 나눠서 모아놓을 수 있다.
    * test runner : 테스트를 돌려주는 것들
  * 기능
    * `setUp(self)` : 테스트 케이스별로 반복되는 행위를 한번에 정의해서 자동으로 호출시킬 수 있다.
    * `tearDown(self)` : 위의 내용을 테스트 끝난 후 자동으로 없앨 수 있다.
    * 테스트 클래스 단위로 중복되는 것 처리할수도 있음
    * `@unittest.skip()` : 일단 테스트 건너뛰는 기능 (잠시만 유지되다가 사라지는 것이 맞다)
    * `@unittest.skipif(조건)` : 조건에 따라서 테스트 건너뛰는 기능
    * `@unittest.expectedFailure` : 우리가 의도한 예외가 잘 수행되는지 확인, 실패가 성공 (negative testcase를 잘 만드는 것이 어렵고, 그 사람의 역량이 드러남)
* Machine Learning Unit Tests
  * 우선 Function 단위가 작아야한다. 그래야 Function 단위로 테스트할 수가 있다.
    * Compose method pattern : Function 내에서 작성되는 내용은 다 같은 레벨에서 작성되어야 한다. -> 중복제거, 재사용성 증가
    * Extract method : Function 들의 기능을 나눠서 만든다.
  * 명확한 문제 : Unit test 하기
  * 명확하지 않은 문제 : 변화가 있는지 측정하라 (ex) 3일동안 학습되는데 뭔가 변화되고 있는지 파악
* Q. 대부분의 오픈소스들에서도 unittest로 테스트되고 관리가 되고 있는 것이 일반적인가요?
  * 그렇다. 일단 오픈소스 다운 받으면 테스트 해봐야 한다.
  * 유닛테스트를 보면 뭘 구현했는지 잘 알 수 있다.

## 학습 회고

* 테스트의 중요성을 알게 되었다. TDD가 익숙해지면 그 전 방식으로 돌아가지 못할 것 같은 예감이 든다.
* unittest 도 template 같은 것들이 있으면 좋을 것 같다.

