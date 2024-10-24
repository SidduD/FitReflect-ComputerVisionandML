# https://www.youtube.com/watch?v=06TE_U21FK4

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

#load the model
with open('shoulder_press.pkl', 'rb') as f:
    model = pickle.load(f)

def calculate_angle(a,b,c):
    a = np.array(a) # first
    b = np.array(b) # mid
    c = np.array(c) # end

    # calculate elbow angle
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle



cap = cv2.VideoCapture(0)

# setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # video feed
    while(cap.isOpened()):
        ret,frame = cap.read()

        # recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # make detection
        results = pose.process(image)
        
        # recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            
            # Calculate angles
            elbow_angle_L = calculate_angle(left_shoulder, left_elbow, left_wrist)
            elbow_angle_R = calculate_angle(right_shoulder, right_elbow, right_wrist)

            shoulder_angle_L = calculate_angle(left_hip, left_shoulder, left_elbow)
            shoulder_angle_R = calculate_angle(right_hip, right_shoulder, right_elbow)

            hip_angle_L = calculate_angle(left_knee, left_hip, left_shoulder)
            hip_angle_R = calculate_angle(right_knee, right_hip, right_shoulder)

            knee_angle_L = calculate_angle(left_ankle, left_knee, left_hip)
            knee_angle_R = calculate_angle(right_ankle, right_knee, right_hip)

            ankle_angle_L = calculate_angle(left_foot_index, left_ankle, left_knee)
            ankle_angle_R = calculate_angle(right_foot_index, right_ankle, right_knee)

            
            # Visualize angles

            # LEFT ELBOW
            cv2.putText(image, 
                        str(elbow_angle_L), 
                        tuple(np.multiply(left_elbow, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )
            
            # LEFT SHOULDER
            cv2.putText(image, 
                        str(shoulder_angle_L), 
                        tuple(np.multiply(left_shoulder, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # LEFT HIP
            cv2.putText(image, 
                        str(hip_angle_L), 
                        tuple(np.multiply(left_hip, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # LEFT KNEE
            cv2.putText(image, 
                        str(knee_angle_L), 
                        tuple(np.multiply(left_knee, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # LEFT ANKLE
            cv2.putText(image, 
                        str(ankle_angle_L), 
                        tuple(np.multiply(left_ankle, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # RIGHT ELBOW
            cv2.putText(image, 
                        str(elbow_angle_R), 
                        tuple(np.multiply(right_elbow, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )   

            # RIGHT SHOULDER
            cv2.putText(image, 
                        str(shoulder_angle_R), 
                        tuple(np.multiply(right_shoulder, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )
            
            # RIGHT HIP
            cv2.putText(image, 
                        str(hip_angle_R), 
                        tuple(np.multiply(right_hip, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # RIGHT KNEE
            cv2.putText(image, 
                        str(knee_angle_R), 
                        tuple(np.multiply(right_knee, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # RIGHT ANKLE
            cv2.putText(image, 
                        str(ankle_angle_R), 
                        tuple(np.multiply(right_ankle, [cap.get(3),cap.get(4)]).astype(int)), # converting coord to actual coord based on camer feed size
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                       )
            
            # evaluate pose with model
            row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

            # make detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    
        except:
            pass
    
        # render detections
        mp_drawing.draw_landmarks(image, 
                                  results.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # joint
                                  mp_drawing.DrawingSpec(color=(245,66,117), thickness=2, circle_radius=2)  # connections
                                  )
        
       
        
        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
   