import cv2


def detection_blur(image):
    # -100.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return 0. - float(cv2.Laplacian(gray, cv2.CV_64F).var())
