import cv2

img_bagic = cv2.imread("cat.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Image_Bagic",img_bagic)
cv2.waitKey(0)
cv2.imwrite("result.png", img_bagic)

cv2.destroyAllWindows()                                # 창을 다 닫고 다음 소스코드 실행

img_gray = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Image_Gray", img_gray)
cv2.waitKey(0)
cv2.imwrite("result2.png", img_gray)

cv2.destroyAllWindows()

# cv2.cvtColor
img_gray2 = cv2.cvtColor(img_bagic, cv2.COLOR_BGR2GRAY)           # BGR -> GRAY로 변환
cv2.imshow("Image_Gray", img_gray2)
cv2.waitKey(0)
cv2.imwrite("result3.png", img_gray2)

