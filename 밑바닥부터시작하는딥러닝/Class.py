# 클래스 구현
class man:
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model
        print("시작한다!")
    def hello(self):
        print(f"hello!{self.model}")
    def goodbye(self, a):
        for _ in range(a):
            print(f"bye!! {self.model}")

# 생성자 정의
m = man("aa", "m1")

m.hello()
m.goodbye(10)

