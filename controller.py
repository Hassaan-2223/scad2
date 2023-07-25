# import pygame
# import cv2
# import numpy as np
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend','openPalm']  # Customize the gesture classes based on your dataset

# # Initialize Pygame
# pygame.init()
# screen = pygame.display.set_mode((640, 480))

# # Load video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create Pygame clock object
# clock = pygame.time.Clock()

# # Load trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Gesture variables
# hand_cascade = cv2.CascadeClassifier('hand.xml')
# is_paused = False

# while True:
#     # Capture video frames and detect hands
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Gesture recognition logic
#         hand_roi = gray[y:y+h, x:x+w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))  # Resize hand ROI to match the input size of the model
#         hand_roi = np.expand_dims(hand_roi, axis=-1)
#         hand_roi = np.expand_dims(hand_roi, axis=0)
#         hand_roi = hand_roi / 255.0  # Normalize pixel values
#         prediction = model.predict(hand_roi)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         if gesture_class == 'fist':
#             # If the gesture is a fist, pause or play the video
#             if is_paused:
#                 pygame.mixer.music.unpause()  # Uncomment this line for audio playback
#                 is_paused = False
#             else:
#                 pygame.mixer.music.pause()  # Uncomment this line for audio playback
#                 is_paused = True

#     # Display video frames
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = pygame.surfarray.make_surface(frame)
#     screen.blit(frame, (0, 0))
#     pygame.display.flip()

#     # Event handling
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             video.release()
#             pygame.quit()
#             exit(0)

#     clock.tick(30)

















# import cv2
# import numpy as np
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load trained model
# model = tf.keras.models.load_model('hand_gesture_model.h5')

# # Gesture variables
# hand_cascade = cv2.CascadeClassifier('hand.xml')
# is_paused = False

# # Load video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# while True:
#     # Capture video frames and detect hands
#     ret, frame = video.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Gesture recognition logic
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))  # Resize hand ROI to match the input size of the model
#         hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
#         hand_roi = np.expand_dims(hand_roi, axis=0)
#         hand_roi = hand_roi / 255.0  # Normalize pixel values
#         prediction = model.predict(hand_roi)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         if gesture_class == 'closedFist':
#             # If the gesture is a closed fist, pause or play the video
#             if is_paused:
#                 video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES) - 1))  # Resume video playback
#                 is_paused = False
#             else:
#                 video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES) + 1))  # Pause video playback
#                 is_paused = True

#     # Display video frames
#     cv2.imshow('Video', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video', 800, 600)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y+h, x:x+w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi = np.expand_dims(hand_roi, axis=-1)
#         hand_roi = np.expand_dims(hand_roi, axis=0)
#         hand_roi = hand_roi / 255.0

#         prediction = model.predict(hand_roi)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             # For example, rewind the video
#             print("Multi-finger bend gesture detected - Rewind the video")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     # Display the processed video frame
#     cv2.imshow('Video', frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video', 800, 600)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y+h, x:x+w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             # For example, rewind the video
#             print("Multi-finger bend gesture detected - Rewind the video")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     # Display the processed video frame
#     cv2.imshow('Video', frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Split the window into two halves
#     width = frame.shape[1]
#     half_width = width // 2
#     left_frame = frame[:, :half_width, :]
#     right_frame = frame[:, half_width:, :]

#     for (x, y, w, h) in hands:
#         cv2.rectangle(right_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             # For example, rewind the video
#             print("Multi-finger bend gesture detected - Rewind the video")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     # Display the left and right frames
#     cv2.imshow('Video', np.concatenate((left_frame, right_frame), axis=1))

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video', 1280, 480)

# # Create a window to display the camera feed
# cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera', 640, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Split the window into two halves
#     width = frame.shape[1]
#     half_width = width // 2
#     left_frame = frame[:, :half_width, :]
#     right_frame = frame[:, half_width:, :]

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         cv2.rectangle(right_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             # For example, rewind the video
#             print("Multi-finger bend gesture detected - Rewind the video")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     # Display the left and right frames
#     cv2.imshow('Video', np.concatenate((left_frame, right_frame), axis=1))

#     # Display the camera feed
#     cv2.imshow('Camera', frame_camera)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and camera, and close the OpenCV windows
# video.release()
# camera.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             # For example, rewind the video
#             print("Multi-finger bend gesture detected - Rewind the video")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     for (x, y, w, h) in hands_camera:
#         # Draw rectangle around hand in the camera frame
#         cv2.rectangle(frame_camera, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Convert hand region to black and white
#         hand_roi_camera = gray_camera[y:y + h, x:x + w]
#         _, hand_roi_camera = cv2.threshold(hand_roi_camera, 127, 255, cv2.THRESH_BINARY)

#         # Add dots to visualize hand gestures
#         dot_size = 5
#         cv2.circle(hand_roi_camera, (int(w / 2), int(h / 2)), dot_size, (255, 255, 255), -1)

#         # Determine if hand is open or closed based on dot presence
#         num_dots = np.sum(hand_roi_camera == 255)
#         if num_dots < dot_size * dot_size:
#             print("Hand is closed")
#         elif num_dots == dot_size * dot_size:
#             print("Hand is open")

#     # Display the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, frame_camera), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and camera, and close the OpenCV windows
# video.release()
# camera.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Resize camera frame to match video frame dimensions
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0] // 2))

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             print("Multi-finger bend gesture detected")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             print("Open palm gesture detected")

#     # Convert the camera frame to grayscale
#     gray_frame_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding to convert the grayscale image to black and white
#     _, bw_frame_camera = cv2.threshold(gray_frame_camera, 127, 255, cv2.THRESH_BINARY)

#     # Draw dots on the camera frame to indicate hand status
#     for (x, y, w, h) in hands_camera:
#         cv2.circle(bw_frame_camera, (x + w // 2, y + h // 2), 5, (255, 255, 255), -1)

#     # Display the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, cv2.cvtColor(bw_frame_camera, cv2.COLOR_GRAY2BGR)), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Resize camera frame to match video frame dimensions
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0] // 2))

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             print("Multi-finger bend gesture detected")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             print("Open palm gesture detected")

#     # Resize the camera frame to match the video frame dimensions
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] - frame.shape[1] // 2, frame.shape[0]))

#     # Apply thresholding to convert the grayscale image to black and white
#     _, bw_frame_camera = cv2.threshold(gray_camera, 127, 255, cv2.THRESH_BINARY)

#     # Draw dots on the camera frame to indicate hand status
#     for (x, y, w, h) in hands_camera:
#         cv2.circle(bw_frame_camera, (x + w // 2, y + h // 2), 5, (255, 255, 255), -1)

#     # Concatenate the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, cv2.cvtColor(bw_frame_camera, cv2.COLOR_GRAY2BGR)), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Resize camera frame to match video frame dimensions
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0] // 2))

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             print("Multi-finger bend gesture detected")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             print("Open palm gesture detected")

#     # Resize the camera frame to match the video frame dimensions
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0]))

#     # Apply thresholding to convert the grayscale image to black and white
#     _, bw_frame_camera = cv2.threshold(gray_camera, 127, 255, cv2.THRESH_BINARY)

#     # Draw dots on the camera frame to indicate hand status
#     for (x, y, w, h) in hands_camera:
#         cv2.circle(bw_frame_camera, (x + w // 2, y + h // 2), 5, (255, 255, 255), -1)

#     # Concatenate the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, cv2.cvtColor(bw_frame_camera, cv2.COLOR_GRAY2BGR)), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
# cv2.destroyAllWindows()









# import cv2
# import numpy as np
# import tensorflow as tf

# # Constants
# GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# # Load the trained model
# model = tf.keras.models.load_model('C:/Users/Hassaan/Desktop/hand video player controller/hand_gesture_model.h5')

# # Gesture variables
# hand_cascade = cv2.CascadeClassifier('C:/Users/Hassaan/Desktop/hand video player controller/hand.xml')

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Resize camera frame to match video frame height
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0]))

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             print("Multi-finger bend gesture detected")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             print("Open palm gesture detected")

#     # Resize the camera frame to match the video frame height
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0]))

#     # Apply thresholding to convert the grayscale image to black and white
#     _, bw_frame_camera = cv2.threshold(gray_camera, 127, 255, cv2.THRESH_BINARY)

#     # Draw dots on the camera frame to indicate hand status
#     for (x, y, w, h) in hands_camera:
#         cv2.circle(bw_frame_camera, (x + w // 2, y + h // 2), 5, (255, 255, 255), -1)

#     # Concatenate the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, cv2.cvtColor(bw_frame_camera, cv2.COLOR_GRAY2BGR)), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
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

# # Initialize the video file
# video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# # Create a window to display the video and hand detection
# cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

# while video.isOpened():
#     # Capture video frames
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform hand detection on the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Open the camera and capture a frame
#     camera = cv2.VideoCapture(0)
#     ret_camera, frame_camera = camera.read()
#     if not ret_camera:
#         break

#     # Resize camera frame to match video frame height
#     frame_camera = cv2.resize(frame_camera, (frame.shape[1] // 2, frame.shape[0]))

#     # Perform hand detection on the camera frame
#     gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
#     hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in hands:
#         # Draw rectangle around hand in the video frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Perform hand recognition and take actions based on gestures
#         hand_roi = gray[y:y + h, x:x + w]
#         hand_roi = cv2.resize(hand_roi, (64, 64))
#         hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

#         hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
#         hand_roi_rgb = hand_roi_rgb / 255.0

#         prediction = model.predict(hand_roi_rgb)
#         gesture_index = np.argmax(prediction)
#         gesture_class = GESTURE_CLASSES[gesture_index]

#         # Perform action based on the recognized gesture
#         if gesture_class == 'closedFist':
#             # Perform action for closed fist gesture
#             # For example, pause the video
#             print("Closed fist gesture detected - Pause the video")

#         elif gesture_class == 'fingerCircle':
#             # Perform action for finger circle gesture
#             # For example, play the video
#             print("Finger circle gesture detected - Play the video")

#         elif gesture_class == 'multiFingerBend':
#             # Perform action for multi-finger bend gesture
#             print("Multi-finger bend gesture detected")

#         elif gesture_class == 'openPalm':
#             # Perform action for open palm gesture
#             # For example, forward the video
#             print("Open palm gesture detected - Forward the video")

#     # Invert the colors of the hand region in the camera frame
#     for (x, y, w, h) in hands_camera:
#         hand_region = gray_camera[y:y + h, x:x + w]
#         hand_region = cv2.resize(hand_region, (64, 64))
#         _, inverted_hand_region = cv2.threshold(hand_region, 127, 255, cv2.THRESH_BINARY_INV)
#         frame_camera[y:y + h, x:x + w] = cv2.cvtColor(inverted_hand_region, cv2.COLOR_GRAY2BGR)

#     # Concatenate the video frame and camera frame side by side
#     display_frame = np.concatenate((frame, frame_camera), axis=1)
#     cv2.imshow('Hand Gesture Detection', display_frame)

#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the OpenCV windows
# video.release()
# cv2.destroyAllWindows()







import cv2
import numpy as np
import tensorflow as tf

# Constants
GESTURE_CLASSES = ['closedFist', 'fingerCircle', 'multiFingerBend', 'openPalm']  # Customize the gesture classes based on your dataset

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Gesture variables
hand_cascade = cv2.CascadeClassifier('hand.xml')

# Initialize the video file
video = cv2.VideoCapture('C:/Users/Hassaan/Desktop/hand video player controller/video/SVID_20200512_070934.mp4')

# Create a window to display the video and hand detection
cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Gesture Detection', 1280, 480)

while video.isOpened():
    # Capture video frames
    ret, frame = video.read()
    if not ret:
        break

    # Perform hand detection on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in hands:
        # Draw rectangle around hand in the video frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Perform hand recognition and take actions based on gestures
        hand_roi = gray[y:y + h, x:x + w]
        hand_roi = cv2.resize(hand_roi, (64, 64))
        hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

        hand_roi_rgb = np.expand_dims(hand_roi_rgb, axis=0)
        hand_roi_rgb = hand_roi_rgb / 255.0

        prediction = model.predict(hand_roi_rgb)
        gesture_index = np.argmax(prediction)
        gesture_class = GESTURE_CLASSES[gesture_index]

        # Perform action based on the recognized gesture
        if gesture_class == 'closedFist':
            # Perform action for closed fist gesture
            # For example, pause the video
            print("Closed fist gesture detected - Pause the video")

        elif gesture_class == 'fingerCircle':
            # Perform action for finger circle gesture
            # For example, play the video
            print("Finger circle gesture detected - Play the video")

        elif gesture_class == 'multiFingerBend':
            # Perform action for multi-finger bend gesture
            print("Multi-finger bend gesture detected")

        elif gesture_class == 'openPalm':
            # Perform action for open palm gesture
            # For example, forward the video
            print("Open palm gesture detected - Forward the video")

    # Display the video frame
    cv2.imshow('Video', frame)

    # Open the camera and capture a frame
    camera = cv2.VideoCapture(0)
    ret_camera, frame_camera = camera.read()
    if not ret_camera:
        break

    # Perform hand detection on the camera frame
    gray_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)
    hands_camera = hand_cascade.detectMultiScale(gray_camera, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in hands_camera:
        # Draw rectangle around hand in the camera frame
        cv2.rectangle(frame_camera, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Invert the colors of the hand region in the camera frame
    for (x, y, w, h) in hands_camera:
        hand_region = gray_camera[y:y + h, x:x + w]
        hand_region = cv2.resize(hand_region, (64, 64))
        _, inverted_hand_region = cv2.threshold(hand_region, 127, 255, cv2.THRESH_BINARY_INV)
        frame_camera[y:y + h, x:x + w] = cv2.cvtColor(inverted_hand_region, cv2.COLOR_GRAY2BGR)

    # Display the camera frame
    cv2.imshow('Camera', frame_camera)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video.release()
cv2.destroyAllWindows()









