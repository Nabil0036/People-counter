import cv2

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    cv2.imshow("Video",frame)

    q = cv2.waitKey(1)
    if q == 27:
        break

cap.release()
cv2.destroyAllWindows()