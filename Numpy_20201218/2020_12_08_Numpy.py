import numpy as np

list_Data = [1, 2, 3, 4]

# numpy로 변환 -> np.array(list데이터), array.size(길이), array.dtype(자료형), array[index] index로 원소값에 접근
array = np.array(list_Data)
print("list를 numpy 배열로 변환!", array)

# 0부터 3까지의 배열을 만들기
array1 = np.arange(4) # np.arange(정수) 0 ~ 정수-1 배열을 만듬
print("np.arange(4) :", array1)

# 2차원 배열 데이터를 만들고, 모든 원소를 0으로 초기화, 그 자료형들은 모두 실수를 가진다.
array2 = np.zeros((3, 3), dtype=float)
print("2차원 배열 데이터를 만들고, 모든 원소를 0으로 초기화, 그 자료형들은 모두 실수를 가진다.\n", array2)

# 2차원 배열 데이터를 만들고, 모든 원소를 1으로 초기화, 그 자료형들은 모두 문자열을 가진다.
array3 = np.ones((4,4), dtype=str)
print("2차원 배열 데이터를 만들고, 모든 원소를 1으로 초기화, 그 자료형들은 모두 문자열을 가진다.\n", array3)

# 0~9까지의 수를 랜덤하게 들어간 2차원 배열을 만들기
array4 = np.random.randint(0,10,(4,4),dtype=int)                     # 0 ~ 10-1(9) 까지
print("0~9까지의 수를 랜덤하게 들어간 2차원 배열을 만들기\n", array4)

# 표준 정규 분포를 가지는 2차원 배열을 만든다.  (0,1) 평균 = 0, 표준편차 = 1
array5 = np.random.normal(0, 1, (3,3))
print("표준 정규 분포를 가지는 2차원 배열을 만든다.  (0,1) 평균 = 0, 표준편차 = 1\n", array5)

# 두 배열을 합치는 함수
array6 = np.array([1, 2, 3])
array7 = np.array([4, 5, 6])
array8 = np.concatenate([array6, array7])
print("두 배열을 합치는 함수\n", array8)
print(array8.shape)

# 배열 사이즈를 변경한다. 1*4 배열을 2*2로 변경해보자.
array9 = np.array([1, 2, 3, 4])
array10 = array9.reshape((2, 2))
print("배열 사이즈를 변경한다. 1*4 배열을 2*2로 변경해보자.\n", array10)

# row 세로축을 기준으로 행렬을 합쳐보자.
array11 = np.array([1, 2, 3, 4]).reshape(1, 4)
array12 = np.array([10, 20, 30, 40, 50, 60, 70, 80]).reshape(2,4)
print("row 세로축을 기준으로 행렬을 합쳐보자.\n", array11)
print(array12)
array13 = np.concatenate([array11, array12], axis=0)
print("row 세로축을 기준으로 행렬을 합쳐보자.\n", array13)

# concatenate([행렬1,행렬2], axis = 0 or axis = 1)
array22 = np.arange(4).reshape(2,2)
array23 = np.arange(4,8).reshape(2,2)
print("array22 :\n", array22)
print("array23 :\n", array23)
array24 = np.concatenate([array22, array23], axis= 1)
print("np.concatenate([array22][array23], axis = 1)\n", array24)
array25 = np.concatenate([array22, array23], axis= 0)
print("np.concatenate([array22][array23], axis = 0)\n", array25)

# 2차원 배열을 left, right로 나누어 보자.
array14 = np.arange(0, 8).reshape(2, 4)
print("2차원 배열을 left, right로 나누어 보자.\n", array14)
left, right = np.split(array14, [1], axis=1)                  # array14의 1번 인덱스, row 기준으로 left, right로 나누어라!
print("왼쪽 :\n", left)
print("오른쪽 :\n", right)
print("왼쪽 크기 :", left.shape)
print("오른쪽 크기 :", right.shape)

# numpy 1차원 배열 연산 더하기, 곱셈
array15 = np.random.randint(0, 10, size=4, dtype=int)
print("array15 =", array15, "\n" ,"array15 * 5 =", array15 * 5, "\narray15 + 5 =", array15 + 5, end="")

# numpy 브로드캐스팅하여 형태가 다른 배열끼리의 연산을 하자. ppt 3page
array16 = np.arange(1, 13).reshape(3, 4)
array17 = np.arange(1, 4).reshape(3,1)
array18 = array16 + array17
print("\narray16 =\n", array16, "\narray17 =\n", array17, "\narray16 + array17 =\n", array18)


# numpy 마스킹 연산 : 각 원소의 값이 어떤 기준으로 True가 되는지 False가 되는지?
array19 = np.arange(16).reshape(4,4)
array20 = array19 < 10
print("\narray :\n", array19, "\narray < 10 :\n", array20)
# 마스킹 연산 활용 방법 : True나 False로 된 배열 자리에 내가 원하는 값을 따로 넣어줄 수 있음. (이미지 처리 후 구분하여 다른 값을 넣을 때 유용하게 사용)
array19[array20] = 100
print("array19[array20] = 100\n", array19)

# 집계함수 : numpy 최대, 최소, 평균(mean, average), 합계
array21 = np.arange(16).reshape(4,4)
print("array21 :\n", array21, "\n최대 :", np.max(array21), "최소 :", np.min(array21), "평균 :", np.mean(array21), "합계 :", np.sum(array21))

# 각각의 행, 열에 대해서만 합계를 구해라.
array26 = np.arange(16).reshape(4,4)
print("주어진 행렬 : array26\n", array26)
array27 = np.sum(array26, axis=1)
print("각각의 행에 대해서만 합계를 구해라.")
print("1행 합계 :", array27[0])
print("2행 합계 :", array27[1])
print("3행 합계 :", array27[2])
print("4행 합계 :", array27[3])
print("각각의 열에 대해서만 합계를 구해라.")
array28 = np.sum(array26, axis=0)
print("1열 합계 :", array28[0])
print("1열 합계 :", array28[1])
print("1열 합계 :", array28[2])
print("1열 합계 :", array28[3])






