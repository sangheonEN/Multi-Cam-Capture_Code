import numpy as np

# 1개의 array를 정의하고 save 한 뒤 load하여 출력해보아라.
print("1개의 객체 정의 save load")
array = np.arange(16).reshape(4,4)
np.save("saved.npy", array)

result = np.load("saved.npy")
print(result)

# 복수 객체를 정의하고 save 한 뒤 load하여 출력하라.
print("복수객체 정의 save load")
array1 = np.arange(0, 10)
array2 = np.arange(10, 20)
np.savez("saved.npz", array1 = array1, array2 = array2)

Load_Data = np.load("saved.npz")
result1 = Load_Data['array1']        # save할때 정의된 array 이름으로 접근해서 불러올 수 있다! '이름명' ''따옴표 잊지말자!
result2 = Load_Data['array2']
print(result1)
print(result2)

# numpy 정렬
# 행/열은 2차원 배열일때만 의미가 있고, numpy는 다차원 배열도 다루기 때문에 array[1차원,2차원,...]으로 이해하시는게 좀 더 정확할 것 같습니다
print("numpy 정렬 array :")
array3 = np.random.randint(1,29, (1,6)) # np.random.randint(1,29, (1,6)) (1,6) 행과 열을 선언해주었으니 2차원 배열이다. 그냥 1차원 배열을 하기 위해서는 np.random.randint(1,29, 6)으로 선언해야함. 또한 1차원 배열의 default는 행 기준임
print(array3)
print("numpy 정렬 오름차순")
array3.sort()
print(array3)
print("numpy 정렬 내림차순")
print(array3[:,::-1])          # 2차원 배열의 내림차순 정렬 (:,::-1)

# numpy 2차원 배열일때 열 기준으로 정렬하기
print("numpy 2차원 배열일때 열 기준으로 정렬하기")
array4 = np.random.randint(1,9,(2,4))
print(array4)
print("열 기준으로 정렬 array : ")
array4.sort(axis=0)                      # 형태 잘 기억하기!
print(array4)

# 균일한 간격으로 데이터를 생성할 때 사용 linspace(시작값, 끝값, 몇개의 데이터가 있을까?)
array5 = np.linspace(0, 10, 5)
print("균일한 간격으로 데이터를 생성할 때 사용 linspace(시작값, 끝값, 몇개의 데이터가 있을까?)")
print(array5)

# 난수의 재연 (실행마다 결과 동일) seed(숫자)
np.random.seed(5)
print(np.random.randint(0, 10, (2,5)))

# Numpy 배열 객채 복사해서 다른 array에 저장되더라도 다른 array 인덱스에 접근해서 값을 변경하지 못하게 함.
print("copy()함수를 쓰지 않으니까 다른 array에서 인덱스로 값 변경이 가능해버린다!")
array6 = np.arange(1,7)
array7 = array6
array7[0] = 100
print(array6)
print("copy()함수를 쓰니까 다른 array에서 인덱스로 값 변경이 불가능해버린다!")
array9 = np.arange(1,7)
array8 = array9.copy()
array8[0] = 100
print(array9)

# Numpy 중복된 원소 제거
print("중복된 원소 제거 np.unique")
array10 = np.array([1, 2, 3, 3, 2, 1, 5, 6, 7, 8])
print(np.unique(array10))
