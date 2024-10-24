import cv2
import mediapipe as mp
import numpy as np

import websockets
import asyncio
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

connected_clients = set()  # Set to keep track of connected clients

def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    # calculate elbow angle
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

async def handle_client(websocket):
    connected_clients.add(websocket)
    print("Client connected.")
    try:
        # Send a welcome message to the client
        await websocket.send("Welcome to the server!")
        while True:
            await asyncio.sleep(1)  # Keep the connection alive
    except websockets.exceptions.ConnectionClosed:
        connected_clients.remove(websocket)

async def send_data():
    # Define the WebSocket server URL
    websocket_server_url = "localhost"
    websocket_server_port = 8765

    # Start the WebSocket server
    async with websockets.serve(handle_client, websocket_server_url, websocket_server_port):
        await asyncio.Future()  # run forever

    print(f"Server started at ws://{websocket_server_url}:{websocket_server_port}")

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None
    prevStage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Video feed
        while(cap.isOpened()):
            ret, frame = cap.read()

            # Recolor image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Curl logic
                if angle > 160:
                    stage = "down"
                   
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(counter)

                if prevStage != stage:
                    data = {
                        "stage": stage,
                        "counter": counter
                    }
                    # Send data to all connected clients
                    for client in connected_clients:
                        print("sendning new vals")
                        await client.send(json.dumps(data))
                prevStage = stage
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (300,73), (245,117,16), -1)

            # Rep data
            cv2.putText(image, "REPS", (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, "STAGE", (105,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, 
                                      results.pose_landmarks, 
                                      mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joint
                                      mp_drawing.DrawingSpec(color=(245,66,117), thickness=2, circle_radius=2)  # Connections
                                      )
            
            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Run the asyncio event loop
asyncio.run(send_data())
