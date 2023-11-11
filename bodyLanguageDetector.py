# https://www.youtube.com/watch?v=We1uB79Ci-w

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_facemesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# setup mediapipe instance
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # video feed
    while(cap.isOpened()):
        ret,frame = cap.read()

        # recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # make detection
        results = holistic.process(image)
        
        # recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(image, 
                                  results.face_landmarks, 
                                  mp_facemesh.FACEMESH_CONTOURS, 
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )

        # Draw Right hand
        mp_drawing.draw_landmarks(image, 
                                  results.right_hand_landmarks, 
                                  mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        # Draw Left hand
        mp_drawing.draw_landmarks(image, 
                                  results.left_hand_landmarks, 
                                  mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        
        # render detections
        mp_drawing.draw_landmarks(image, 
                                  results.pose_landmarks, 
                                  mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
        
        # try:
           
        # except:
        #     pass
        
        
        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()