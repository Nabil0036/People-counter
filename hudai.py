import cv2
from imutils.video import FPS
from imutils.video import WebcamVideoStream as webcam
cap = cv2.VideoCapture(1)
fps = FPS().start()

# vs = webcam(src=0).start()
while True:
    fps.update()
    _,frame = cap.read()
    # frame = vs.read()
    cv2.imshow("Video",frame)

    q = cv2.waitKey(1)
    if q == 27:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()