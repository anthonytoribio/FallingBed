import cv2
import numpy as np

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def getBedBoundingBox() -> tuple:
    cap = cv2.VideoCapture(0)
    brightest_rectangle = None
    totalFrames = 0
    canvas = None
    while totalFrames < 15:
        ret, img = cap.read()
        # HSV thresholding to get rid of as much background as possible
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([10, 0, 20])
        upper_blue = np.array([150, 40, 225]) 
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        b, g, r = cv2.split(result)
        g = clahe(g, 5, (3, 3))

        # Adaptive Thresholding to isolate the bed
        img_blur = cv2.blur(g, (9, 9))
        img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 2)

        contours, hierarchy = cv2.findContours(img_th,
                                                cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)

        # Filter the rectangle by choosing only the big ones
        # and choose the brightest rectangle as the bed
        max_brightness = 0
        canvas = img.copy()
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            if w*h <= 50000 and w*h > 25000:
                mask = np.zeros(img.shape, np.uint8)
                mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
                brightness = np.sum(mask)
                if brightness > max_brightness:
                    brightest_rectangle = rect
                    max_brightness = brightness
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)
        totalFrames += 1

    x, y, w, h = brightest_rectangle
    print(x,y,w,h)
    cv2.rectangle(canvas, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("canvas", canvas)
    cv2.imwrite("samples/bedD.jpg", canvas)
    return brightest_rectangle

if __name__ == "__main__":
    getBedBoundingBox()

