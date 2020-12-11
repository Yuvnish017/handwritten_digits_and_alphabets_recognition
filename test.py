import cv2 as cv
import numpy
import imutils
from imutils.contours import sort_contours
from keras.models import load_model

model = load_model('final_model.h5')
image = cv.imread('final_test.jpg')
image = cv.resize(image, (800, 800))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

edged = cv.Canny(gray, 30, 150)
contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method="left-to-right")[0]

for c in contours:
    (x, y, w, h) = cv.boundingRect(c)
    if 25 <= w <= 500 and 25 <= h <= 500:
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
            cv.putText(image, "A", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 1:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "B", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 2:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "C", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 3:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "D", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 4:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "E", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 5:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "F", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 6:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "G", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 7:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "H", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 8:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "I", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 9:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "J", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 10:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "K", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 11:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "L", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 12:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "M", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 13:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "N", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 14:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "O", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 15:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "P", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 16:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "Q", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 17:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "R", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 18:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "S", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 19:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "T", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 20:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "U", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 21:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "V", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 22:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "W", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 23:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "X", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 24:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "Y", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 25:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "Z", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 26:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "0", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 27:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "1", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 28:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "2", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 29:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "3", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 30:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "4", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 31:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "5", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 32:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "6", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 33:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "7", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 34:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "8", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        if z == 35:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(image, "9", (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

cv.imshow('image', image)
cv.imwrite('final_output.jpg', image)
cv.waitKey(0)
cv.destroyAllWindows()
