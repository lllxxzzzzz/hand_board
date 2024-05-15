import cv2


# 读取图像
image = cv2.imread('C:\\Users\\24223\\Desktop\\q.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('1', image)
# 将像素值100以上的设置为255，100以下的设置为0
threshold_value =150
max_value = 255
_, thresholded_image = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY)

# 显示结果
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
