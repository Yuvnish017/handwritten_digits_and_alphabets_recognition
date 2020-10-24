import cv2 as cv
import numpy
import imutils
from imutils.contours import sort_contours
from keras.models import load_model

model = load_model('digits_v1.h5')
image = cv.imread('img.jpg')
image = cv.resize(image, (1000, 1000))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

edged = cv.Canny(gray, 30, 150)
contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method="left-to-right")[0]

for c in contours:
    (x, y, w, h) = cv.boundingRect(c)
    if 25 <= w <= 150 and 25 <= h <= 100:
        roi = gray[y:y+h, x:x+w]
        thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        (th, tw) = thresh.shape
        if tw > th:
            thresh = imutils.resize(thresh, width=32)
        if th > tw:
            thresh = imutils.resize(thresh, height=32)
        (th, tw) = thresh.shape
        dx = int(max(0, 32 - tw)/2.0)
        dy = int(max(0, 32 - th) / 2.0)
        padded = cv.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        padded = cv.resize(padded, (32, 32))
        padded = numpy.array(padded)
        padded = numpy.expand_dims(padded, axis=0)
        prediction = model.predict(padded)*100
        max_value = max(prediction[0])
        output_array = [1 if i >= max_value else 0 for i in prediction[0]]
        z = output_array.index(1)
        if z == 0:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "0", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 1:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "1", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 2:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "2", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 3:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "3", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 4:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "4", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 5:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "5", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 6:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "6", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 7:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "7", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 8:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "8", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 9:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "9", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()
