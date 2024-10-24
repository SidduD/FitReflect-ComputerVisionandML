
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

from landmarks import landmarks

import websockets
import asyncio
import json


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

async def send_data():
    # Define the WebSocket server URL
    websocket_server_url = "ws://localhost:8765"

    # Connect to the WebSocket server
    async with websockets.connect(websocket_server_url) as websocket:

        with open('squat.pkl', 'rb') as f:
            model1 = pickle.load(f)

        with open('depth.pkl', 'rb') as f:
            model2 = pickle.load(f)

        with open('knees.pkl', 'rb') as f:
            model3 = pickle.load(f)
        
        cap = cv2.VideoCapture(0)
        counter = 0 
        current_stage = ''
      
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

                  # render detections
                mp_drawing.draw_landmarks(image, 
                                          results.pose_landmarks, 
                                          mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # joint
                                          mp_drawing.DrawingSpec(color=(245,66,117), thickness=2, circle_radius=2)  # connections
                                          )
                # extract landmarks
                try:
                    row = np.array([[res.x,res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                    X = pd.DataFrame([row], columns=landmarks)
                    body_language_class = model1.predict(X)[0]
                    body_language_prob = model1.predict_proba(X)[0]
                    
                    depth_class = model2.predict(X)[0]
                    knees_class = model3.predict(X)[0]

                    if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.55:
                        current_stage = 'down'
                    elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.55:
                        current_stage = 'up'
                        counter +=1
                        data = {
                            "source": "ML-model",
                            "destination": "counters",
                            "stage": current_stage,
                            "counter": counter
                        }
                        await websocket.send(json.dumps(data))

                    if body_language_class == 'down' and depth_class == 'high':
                        depth = 'high' 
                    elif body_language_class == 'down' and depth_class == 'ideal':
                        depth = 'ideal'
                    else:
                        depth = '-' 
                
                    if body_language_class == 'down' and depth_class == 'ideal' and knees_class == 'cave-in':
                        knees = 'cave-in' 
                    elif body_language_class == 'down' and depth_class == 'ideal' and knees_class == 'ideal':
                        knees = 'ideal'
                    else:
                        knees = '-' 
                    
                    #Get status box
                    cv2.rectangle(image, (0,0), (600,60), (245,117,16), -1)

                    #Display Class
                    cv2.putText(image, 'CLASS'
                                , (300,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (295,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)
                
                    #Display Lean
                    cv2.putText(image, 'KNEES'
                                , (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    # cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2))
                    cv2.putText(image, knees
                                , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                    #Display Stance
                    cv2.putText(image, 'DEPTH'
                                , (180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    #cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2))
                    cv2.putText(image, depth
                                , (175,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                    #Display Counter
                    cv2.putText(image,'COUNT'
                            , (460,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter)
                            , (475,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2, cv2.LINE_AA)
                    
                    landmark = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]

                    # Define the connections between landmarks
                    connections = mp_pose.POSE_CONNECTIONS

                    # # Send landmark data via WebSocket
                    # data = {
                    #     "source": "ML-model",
                    #     "destination": "pose-display",
                    #     "landmarks": landmark,
                    #     "connections": list(connections)
                    # }
                    # await websocket.send(json.dumps(data))
                except:
                    pass
                
                cv2.imshow("Mediapipe Feed", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            await websocket.close()        
        cap.release()
        cv2.destroyAllWindows()
            
asyncio.get_event_loop().run_until_complete(send_data())