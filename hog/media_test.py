# import the necessary packages
import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# test images
cap = cv2.imread('media/bed8.jpg')
# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)
    # resizing for faster detection
    # cap = cv2.resize(cap, (100, 300))
    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(cap, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(cap, (xA+80, yA), (xB-80, yB),
                          (0, 255, 0), 1)
    
    # Write the output video s
    out.write(cap.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',cap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)