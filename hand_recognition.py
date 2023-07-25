#Step -1
import cv2
import numpy as np 
import math
import pyautogui as p
import time as t
     
            

#Read Camera
 cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
def nothing(x):  
    pass
#window name
cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300)) 
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

#COlor Detection Track

cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)


while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(600,500))
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (0,1), (300,500), (255, 0, 0), 0)
    crop_image = frame[1:500, 0:300]
    
    #Step -2
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    #detecting hand
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    #Step -3
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    
    #Step - 4
    #Creating Mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #filter mask with image
    filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
    
    #Step - 5
    mask1  = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments") #getting track bar value
    ret,thresh = cv2.threshold(mask1,m_g,255,cv2.THRESH_BINARY)
    dilata = cv2.dilate(thresh,(3,3),iterations = 6)
    
    #Step -6
    #findcontour(img,contour_retrival_mode,method)
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    try:
        #print("try")
         #Step -7
         # Find contour with maximum area
        cm = max(cnts, key=lambda x: cv2.contourArea(x))
        #print("C==",cnts)
        epsilon = 0.0005*cv2.arcLength(cm,True)
        data= cv2.approxPolyDP(cm,epsilon,True)
    
        hull = cv2.convexHull(cm)
        
        cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
        cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)
        
        #Step - 8
        # Find convexity defects
        hull = cv2.convexHull(cm, returnPoints=False)
        defects = cv2.convexityDefects(cm, hull)
        count_defects = 0
        #print("Area==",cv2.contourArea(hull) - cv2.contourArea(cm))
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
           
            start = tuple(cm[s][0])
            end = tuple(cm[e][0])
            far = tuple(cm[f][0])
            #Cosin Rule
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            #print(angle)
            # if angle <= 50 draw a circle at the far point
            if angle <= 50:
                count_defects += 1
                cv2.circle(crop_image,far,5,[255,255,255],-1)
        
        print("count==",count_defects)
        
        #Step - 9 
        # Print number of fingers
        if count_defects == 0:
            
            cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        elif count_defects == 1:
            
            p.press("space")
            cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 2:
            p.press("up")
            
            cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 3:
            p.press("down")
            
            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 4:
            p.press("right")
            
            cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            pass
           
    except:
        pass
    #step -10    
    cv2.imshow("Thresh", thresh)
    #cv2.imshow("mask==",mask)
    cv2.imshow("filter==",filtr)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) &0xFF    
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()
    















































































































































































































# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Load the trained model for gesture recognition
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Define the classes of gestures in your model
# GESTURE_CLASSES = ['closedFist', 'openPalm']

# # Read Camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# def preprocess_image(image):
#     # Preprocess the image as required by your model
#     # Replace this code with your own preprocessing logic
#     # Convert the image to grayscalec
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize the image to the desired input size of your model
#     resized = cv2.resize(gray, (224, 224))
#     # Normalize the image values between 0 and 1
#     normalized = resized / 255.0
#     # Return the preprocessed image
#     return normalized

# def nothing(x):
#     pass

# # Window name
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

# # Color Detection Track
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

# while True:
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     # Get hand data from the rectangle sub window
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     # Step -2
#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     # Detecting hand
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     # Step -3
#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     # Step -4
#     # Creating Mask
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     # Filter mask with image
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)

#     # Step -5
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments") # Getting track bar value
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     # Step -6
#     # Find contour(img, contour_retrival_mode, method)
#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         # Step -7
#         # Find contour with maximum area
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         # Step -8
#         # Find convexity defects
#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]

#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])

#             # Cosine Rule
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

#             # If angle <= 50 draw a circle at the far point
#             if angle <= 30:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         print("count==", count_defects)

#         # Step -9
#         # Classify the gesture using the trained model
#         gesture_img = preprocess_image(crop_image)  # Preprocess the image as required by the model
#         gesture_img = np.expand_dims(gesture_img, axis=0)  # Add batch dimension
#         prediction = model.predict(gesture_img)  # Make prediction using the model
#         gesture_class = GESTURE_CLASSES[np.argmax(prediction)]  # Get the predicted gesture class

#         # Step -10
#         # Perform actions based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform actions for closed fist gesture
#             p.press("space")
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif gesture_class == 'openPalm':
#             # Perform actions for palm gesture
#             p.press("up")
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         else:
#             pass

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("filter==", filtr)
#     cv2.imshow("Result", frame)

#     key = cv2.waitKey(25) & 0xFF
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
   





















# import cv2
# import numpy as np
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Gesture variables
# hand_cascade = cv2.CascadeClassifier('hand.xml')

# # Open the camera
# camera = cv2.VideoCapture(0)

# while True:
#     # Read frame from the camera
#     ret, frame = camera.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale for hand detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform hand detection
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw a bounding box around the hand
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Get the ROI of the hand
#         hand_roi = gray[y:y + h, x:x + w]

#         # Resize and preprocess the hand ROI for gesture recognition
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)
#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         # Perform gesture recognition on the hand ROI
#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_symbol = GESTURE_CLASSES[gesture_index]

#         # Display the symbol name on the bounding box
#         cv2.putText(frame, gesture_symbol, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Hand Detection', frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the OpenCV window
# camera.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]

#         cv2.putText(frame, gesture_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Color Adjustments", filtr)
#     cv2.imshow("Gesture", frame)
#     all_image = np.hstack((crop_image, frame))
#     cv2.imshow('Contours', all_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]

#         cv2.putText(frame, gesture_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Color Adjustments", filtr)
#     cv2.imshow("Gesture", frame)
#     crop_image_resized = cv2.resize(crop_image, (frame.shape[1], frame.shape[0]))
#     all_image = np.hstack((crop_image_resized, frame))
#     cv2.imshow('Contours', all_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# recognized_gesture = ""

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]
#         recognized_gesture = gesture_class

#     except:
#         pass

#     cv2.putText(frame, recognized_gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Color Adjustments", filtr)
#     cv2.imshow("Gesture", frame)
#     all_image = np.hstack((crop_image, frame))
#     cv2.imshow('Contours', all_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]

#         cv2.putText(frame, gesture_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Masked", mask1)
#     cv2.imshow("Result", frame)

#     all_image = np.hstack((cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB), frame))
#     cv2.imshow('Recognizing Gesture', cv2.resize(all_image, (900, 400)))

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]

#         cv2.putText(crop_image, gesture_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Color Adjustments", filtr)
#     cv2.imshow("Gesture", frame)
#     crop_image_resized = cv2.resize(crop_image, (frame.shape[1], frame.shape[0]))
#     all_image = np.hstack((crop_image_resized, frame))
#     cv2.imshow('Contours', all_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import math
# import pyautogui as p
# import time as t
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Read camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Create trackbars for color adjustments
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300))
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 179, lambda x: None)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, lambda x: None)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 179, 179, lambda x: None)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, lambda x: None)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, lambda x: None)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]

#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)

#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     try:
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)

#         hull = cv2.convexHull(cm)

#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0

#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)

#         if count_defects == 0:
#             cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif count_defects == 1:
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("space")
#         elif count_defects == 2:
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("up")
#         elif count_defects == 3:
#             cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("down")
#         elif count_defects == 4:
#             cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#             p.press("right")
#         else:
#             pass

#         # Gesture Recognition
#         gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, (128, 128))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 128, 128, 1))
#         result = model.predict(reshaped)

#         prediction = np.argmax(result)
#         gesture_class = GESTURE_CLASSES[prediction]

#         cv2.putText(crop_image, gesture_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     except:
#         pass

#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Color Adjustments", filtr)
#     cv2.imshow("Gesture", frame)
#     crop_image_resized = cv2.resize(crop_image, (frame.shape[1], frame.shape[0]))
#     all_image = np.hstack((crop_image_resized, frame))
#     cv2.imshow('Contours', all_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


















# import cv2
# import numpy as np 
# import math
# import pyautogui as p
# import time as t

# # Load the trained model for gesture recognition
# model = load_model('C:\Users\Hassaan\Desktop\hand video player controller\hand_gesture_model.h5')

# # Define the classes of gestures in your model
# GESTURE_CLASSES = ['Closed Fist', 'Palm']

# # Read Camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# def nothing(x):
#     pass

# # Window name
# cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Adjustments", (300, 300)) 
# cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

# # Color Detection Track
# cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
# cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
# cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
# cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

# while True:
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 2)
#     frame = cv2.resize(frame, (600, 500))
#     # Get hand data from the rectangle sub window
#     cv2.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
#     crop_image = frame[1:500, 0:300]
    
#     # Step -2
#     hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
#     # Detecting hand
#     l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
#     l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
#     l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

#     u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
#     u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
#     u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    
    

#     # Step -3
#     lower_bound = np.array([l_h, l_s, l_v])
#     upper_bound = np.array([u_h, u_s, u_v])
    
#     # Step -4
#     # Creating Mask
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     # Filter mask with image
#     filtr = cv2.bitwise_and(crop_image, crop_image, mask=mask)
    
#     # Step -5
#     mask1 = cv2.bitwise_not(mask)
#     m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments") # Getting track bar value
#     ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
#     dilata = cv2.dilate(thresh, (3, 3), iterations=6)
    
#     # Step -6
#     # Find contour(img, contour_retrival_mode, method)
#     cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     try:
#         # Step -7
#         # Find contour with maximum area
#         cm = max(cnts, key=lambda x: cv2.contourArea(x))
#         epsilon = 0.0005 * cv2.arcLength(cm, True)
#         data = cv2.approxPolyDP(cm, epsilon, True)
    
#         hull = cv2.convexHull(cm)
        
#         cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
#         cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)
        
#         # Step -8
#         # Find convexity defects
#         hull = cv2.convexHull(cm, returnPoints=False)
#         defects = cv2.convexityDefects(cm, hull)
#         count_defects = 0
        
#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
           
#             start = tuple(cm[s][0])
#             end = tuple(cm[e][0])
#             far = tuple(cm[f][0])
            
#             # Cosine Rule
#             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            
#             # If angle <= 50 draw a circle at the far point
#             if angle <= 50:
#                 count_defects += 1
#                 cv2.circle(crop_image, far, 5, [255, 255, 255], -1)
        
#         print("count==", count_defects)
        
#         # Step -9
#         # Classify the gesture using the trained model
#         gesture_img = preprocess_image(crop_image)  # Preprocess the image as required by the model
#         gesture_img = np.expand_dims(gesture_img, axis=0)  # Add batch dimension
#         prediction = model.predict(gesture_img)  # Make prediction using the model
#         gesture_class = GESTURE_CLASSES[np.argmax(prediction)]  # Get the predicted gesture class
        
#         # Step -10
#         # Perform actions based on the recognized gesture
#         if gesture_class == 'Closed Fist':
#             # Perform actions for closed fist gesture
#             p.press("space")
#             cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         elif gesture_class == 'Palm':
#             # Perform actions for palm gesture
#             p.press("up")
#             cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#         else:
#             pass
           
#     except:
#         pass
    
#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("filter==", filtr)
#     cv2.imshow("Result", frame)

#     key = cv2.waitKey(25) & 0xFF    
#     if key == 27: 
#         break

# cap.release()
# cv2.destroyAllWindows()







