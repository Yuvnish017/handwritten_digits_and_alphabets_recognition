import cv2 as cv

image = cv.imread('Sreeja_5-9.jpeg')
image = cv.resize(image, (900, 900))
cv.imshow('image', image)
a = 5500
for i in range(25):
    roi = image[36*i:36*(i+1), :, :]
    cv.imwrite(str(a) + '.jpg', roi)
    a += 1
b = 5550
for j in range(5500, 5525):
    img = cv.imread(str(j) + '.jpg')
    for i in range(18):
        roi = img[:, 50*i:50*(i+1), :]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

        cv.imwrite(str(b) + '.jpg', thresh)
        b += 1


cv.waitKey(0)
cv.destroyAllWindows()
