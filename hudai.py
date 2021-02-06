import cv2
from imutils.video import FPS
from imutils.video import WebcamVideoStream as webcam
# cap = cv2.VideoCapture(1)
vs_1 = webcam(0).start()
vs_2 = webcam(1).start()
fps = FPS().start()

# vs = webcam(src=0).start()
while True:
    fps.update()
    # _,frame = cap.read()
    frame_1 = vs_1.read()
    frame_2 = vs_2.read()
    cv2.imshow("Video_1",frame_1)
    cv2.imshow("Video_2",frame_2)

    q = cv2.waitKey(1)
    if q == 27:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()