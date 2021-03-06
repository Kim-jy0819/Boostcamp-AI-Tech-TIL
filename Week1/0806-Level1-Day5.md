# 부스트캠프 5일차

- [부스트캠프 5일차](#부스트캠프-5일차)
  - [오늘 일정 정리](#오늘-일정-정리)
  - [강의 복습 내용](#강의-복습-내용)
    - [[6강] numpy](#6강-numpy)
    - [[7강] pandas](#7강-pandas)
  - [피어 세션 정리](#피어-세션-정리)
  - [오피스아워 & 과제 풀이](#오피스아워--과제-풀이)
  - [학습 회고](#학습-회고)

## 오늘 일정 정리

* 8/6 (금)
  - [x] 파이썬 6,7강 듣기 : numpy, pandas
  - [x] 오피스아워 (Python&AI Math 멘토 팀) (8/6(금) 18:00~19:30) : 필수 과제 해설

## 강의 복습 내용

### [6강] numpy

* ndarray 객체
  * C의 Array를 사용하여 배열 생성 (dynamic typing을 지원하지 않음)
  * properties
    * dtype : 데이터의 타입
    * shape : dimension 구성
      * rank 0 : scalar / rank 1 : vector / rank 2 : matrix / rank n : n-tensor
    * ndim : number of dimensions (rank의 개수)
    * size : data의 개수 (element의 개수)
    * nbytes : 용량

* Handling shape
  * reshape : shape의 크기를 변경, element의 갯수(size)와 순서는 동일
    * -1 : size를 기반으로 개수 선정
    ```python
    a = np.array([5, 6])
    a.reshape(-1, 2) # array([[5, 6]])
    a[np.newaxis, :] # array([[5, 6]])
    ```
  * flatten : 1차원으로 변환

* indexing & slicing
  * `a[0][0]` 또는 `a[0, 0]`
  * `a[1, :2]` 와 `a[1:2, :2]` 의 dimension, shape이 달라진다.
    ```python
    a = np.array([[1, 2, 5, 8],
                  	  [1, 2, 5, 8],
                 	  [1, 2, 5, 8],
                 	  [1, 2, 5, 8]])
    a[1:2, :2].shape # (1, 2)
    a[1, :2].shape # (2,)
    ```

* creation function
  * arange
    * `np.arange(끝)`
    * `np.arange(시작, 끝, step)`
  * ones, zeros and empty
    * `np.zeros(shape, dtype, order)` : shape은 튜플값으로 넣기
    * `np.empty` : shape만 주어지고 빈 ndarray 생성 (메모리 초기화 안됨)
    * `np.ones_like(test_matrix)` : 기존 ndarray의 shape 크기만큼의 ndarray 반환
  * identity : 단위행렬 생성
    * `np.identity(n)` : n은 number of rows
  * eye : 대각선이 1인 행렬
    * `np.eye(N=3, M=5, k=2)` : 시작 인덱스를 k로 변경할 수 있다.
  * diag : 대각 행렬 값을 추출
    * `np.diag(matrix)`
  * random sampling : 각 분포의 모수와 size를 인자로 넣는다.
    * `np.random.uniform(low, high, size)` : 균등 분포
    * `np.random.normal(loc=mean, scale=std, size)` : 정규 분포

* operation functions
  * axis : 모든 operation function을 실행할 때 기준이 되는 dimension 축
  * sum, mean, std 외에도 지수함수, 삼각함수, 하이퍼볼릭 함수 등 수학 연산자 제공
  * concatenate : numpy array를 붙이는 함수
    ```python
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    np.vstack((a, b)) # array([[1, 2, 3], [2, 3, 4]])
    np.hstack((a, b)) # array([1, 2, 3, 2, 3, 4])
    
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    np.vstack((a, b)) # array([[1], [2], [3], [4], [5], [6]])
    np.hstack((a, b)) # array([[1, 2], [2, 3], [3, 4]])
    ```
    * `np.concatenate((a, b), axis)` 

* array operation
  * element-wise operations : shape이 같은 배열 간 연산, 기본적인 사칙연산 지원
  * broadcasting : shape이 다른 배열 간 연산 지원 (주의 필요)
  * dot product : Matrix의 기본 연산
  * transpose : 전치행렬 반환

* comparisons
  * 배열의 크기가 동일할 때, element간 비교(`>, ==, <`)의 결과를 Boolean type으로 반환
  * `np.all(a < 10)`, `np.any(a < 10)` : 각각 and, or 조건에 따른 boolean 값 반환
  * `np.logical_and()`, `np.logical_not()`, `np.logical_or()`
  * `np.isnan(a)`, `np.isfinite(a)`
  * `np.where(condition)` : index 값 반환
    ```python
    a = np.array([1, 2, 3, 4, 5])
    np.where(a > 2) # (array([2, 3, 4]),)
    ```
  * `np.where(condition, TRUE, FALSE)` : condition에 따라 True일 때의 값과 False일 때의 값을 넣을 수 있다.
    ```python
    a = np.array([1, 2, 3, 4, 5])
    np.where(a > 2, 2, a) # array([1, 2, 2, 2, 2])
    ```
  * `np.argmax(a, axis)`, `np.argmin(a, axis)` : 최대값 또는 최소값의 index 반환
  * `np.argsort()` : 값을 정렬한 인덱스 값을 반환

* boolean & fancy index
  * boolean index
    ```python
    condition = a > 3
    a[condition]
    ```
  * fancy index
    ```python
    a = np.array([2, 4, 6, 8])
    b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer
    a[b] # array([2, 2, 4, 8, 6, 4])
    a.take(b) # 위와 동일
    
    a = np.array([[1, 4], [9, 16]])
    b = np.array([0, 0, 1, 1, 0], int)
    c = np.array([0, 1, 1, 1, 1], int)
    a[b, c] # array([1, 4, 16, 16, 4])
    ```
* numpy data i/o
  * `np.loadtxt()` & `np.savetxt()` : text type의 데이터
  * `np.load()` & `np.save()` : npy(numpy object)로 저장 및 로드

### [7강] pandas

* 특징
  * numpy와 통합하여 사용하기 쉬워짐
  * Tabular 데이터 처리하는 데에 사용

* 용어
  * Series : DataFrame 중 하나의 Column에 해당하는 데이터의 모음 Object
    * numpy.ndarray의 subclass
    * index, values 를 가지고 있음
  * DataFrame : Data Table 전체를 포함하는 Object
    * numpy array-like
    * index, columns 를 가지고 있음
    * 각 컬럼은 다른 데이터 타입 가능, 컬럼 삽입/삭제 가능

* 기능
  * indexing : `loc`는 index 이름, `iloc`은 index number
  * `lambda`, `map`, `replace`, `apply`, `applymap` 등의 함수 사용 가능
  * pandas built-in functions : `describe`, `unique`, `sum`, `isnull`, `sort_values`, `corr`, `cor`, `corrwith`, `value_counts` 등
  * Groupby : 그룹별로 모아진 DataFrame 반환
    * `df.groupby(["Team", "Year"])["Points"].sum()` : 두 개의 컬럼으로 groupby 할 경우, index가 두 개 생성(Hierarchical index)
    * grouped : Aggregation(`agg`), Transformation(`transform`), Filtration(`filter`) 가능
      ```python
      grouped = df.groupby("Team") # generator 형태 반환
      for name, group in grouped: # Tuple 형태로 그룹의 key, value 값이 추출됨
          print(name) # key : Team name
          print(group) # value : DataFrame 형태
      ```
  * Pivot Table : index 축은 groupby와 동일, column에 라벨링 값 추가 (엑셀과 비슷)
    ```python
    df.pivot_table(values=["duration"],
                      index=[df.month, df.item], # index
                      columns=df.network, # columns
                      aggfunc="sum",
                      fill_value=0)
    ```
  * Crosstab : 주로 네트워크 형태의 데이터 다룰 때
    ```python
    pd.crosstab(index=df_movie.critic, 
                    columns=df_movie.title, 
                    values=df_movie.rating,
                    aggfunc="first").fillna(0)
    ```
  * Merge : SQL에서 많이 사용하는 Merge와 같은 기능
    * INNER JOIN / LEFT JOIN / RIGHT JOIN / FULL JOIN
  * Concat : 같은 형태의 데이터를 붙이는 연산작업
  * Persistence
    * Database connection
    * XLS persistence : openpyxls 또는 XlsxWrite 사용
    * pickle 파일 형태로 저장

## 피어 세션 정리

* 과제 코드 리뷰 후 배운 점 (Assignment4)
  * if문을 너무 남발하지 말아야겠다.
  * 코드 수를 줄임으로써 가독성이 떨어진다면 그냥 길게 짜자
  * flag를 통한 코드흐름제어도 좋지만, 적절한 loop로 나눠도 된다는 것을 깨달았다.

* 과제 질문
  * Q. 선택과제 3번 무슨 그래프를 그리는게 맞을까요?
    * N(x ; mu=−1) 이라는 확률분포에서 <!-- $x_0=1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_0%3D1">이 나올 가능도(확률밀도)는 [빈칸] 이다."
    * 그러니까 가능도가 결국 확률밀도인 것인가요? 결국 그려야 하는 그래프가 확률밀도함수 인가요?
    * 샘플이 하나여서 그런건지? (+) 그럼 로그도 안 씌워도 되는건지?
  * 답변) 확률밀도함수를 그리는 것을 통해서 모수 <!-- $\mu$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cmu">​​​ 에 따른 가능도 차이를 보여주는 것 같아요. 로그를 취하는건 미분이 더 쉽기 때문임!

## 오피스아워 & 과제 풀이

* 텍스트 다루기(필수과제2,3,4,5)
  * [파이썬 문자열 메서드 문서](https://docs.python.org/ko/3/library/stdtypes.html#string-methods)
    * `capitalize()` : 첫 문자가 대문자이고 나머지가 소문자인 문자열의 복사본을 돌려줍니다.
    * `isdigit()` : 문자열 내의 모든 문자가 디짓이고, 적어도 하나의 문자가 존재하는 경우 True를 돌려주고, 그렇지 않으면 False를 돌려줍니다.
  * [정규표현식 re 라이브러리 문서](https://docs.python.org/ko/3/library/re.html#regular-expression-objects)

* RNN Backpropagation(선택과제2)
  * **w_x에 대한 gradient값들을 더해주는 이유** : w_x가 s1...sn을 만드는데에 관여를 했고(계산그래프가 그려졌고), 이 때 **copy gate**를 통해서 여러 연산에 참여할 수 있었다고 보면, 계산 그래프의 역방향으로 들어온 gradient를 모두 합(sum)해주는게 맞는 것 같다.
    * <img src="https://user-images.githubusercontent.com/35680202/128513513-816494ea-a683-4f2e-b5f1-6b50612b2ee3.png" width="550" height="300">
    * <img src="https://user-images.githubusercontent.com/35680202/128593331-d174c719-12b1-49f2-ba60-309ae96b0631.png" width="550" height="300">

## 학습 회고
* 매일 학습 정리를 작성하니까 힘들지 않고 좋은 것 같다.
* 평일에 놓쳤던 디테일한 부분들을 주말에 채우도록 노력해야겠다.
* 뭔가 파이썬 잘 안다고 생각했는데, lambda, map 등을 자유자재로 쓸 수 있을 정도로 연습을 더 해야겠다는 생각이 들었다.
