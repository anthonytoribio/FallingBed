import cv2
import numpy as np

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def main():
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    # Get Bounding box for the bed
    cap = cv2.VideoCapture('res/IMG_0951.MOV')
    brightest_rectangle = None
    totalFrames = 0
    canvas = None
    # Find the bed within 18 frames
    while totalFrames < 18:
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

        # Filter the rectangle by choosing only the big ones (within a certain area)
        # and choose the brightest rectangle as the bed
        # also filter with rectangle not starting at the edges of the frame
        max_brightness = 0
        canvas = img.copy()
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            if w*h <= 1000000 and w*h > 25000 and x != 0 and y != 0:
                mask = np.zeros(img.shape, np.uint8)
                mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
                brightness = np.sum(mask)
                if brightness > max_brightness and (not brightest_rectangle or 
                                                    w*h > brightest_rectangle[2] * brightest_rectangle[3]):
                    brightest_rectangle = rect
                    max_brightness = brightness
        totalFrames += 1

    x, y, w, h = brightest_rectangle
    cv2.rectangle(canvas, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("canvas", canvas)
    cv2.imwrite("samples/bedD.jpg", canvas)
    print(f"FOUND BED BOUNDING BOX: {brightest_rectangle}")

    # Get the dynamic bounding box of user
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # # resizing for faster detection
        # frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,2))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        x,y,w,h = brightest_rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 3)

        for (xA, yA, xB, yB) in boxes:
            # Check if the user's bounding box is outside the bed's bounding box
            # display the detected boxes in the colour picture with either green or red
            if (xA+150 < x or yA < y or xB-150 > x+w or yB > y+h):
                cv2.rectangle(frame, (xA+150, yA), (xB-150, yB),
                            (0, 0, 255), 2)
                #TODO: create a buzzing
            else:
                cv2.rectangle(frame, (xA+150, yA), (xB-150, yB),
                            (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()


if __name__=="__main__":
    main()