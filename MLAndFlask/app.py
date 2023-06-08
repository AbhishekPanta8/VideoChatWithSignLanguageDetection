from flask import Flask, request
from flask_socketio import SocketIO, emit, send
import cv2
import io
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions =(['hello','goodbye','please','no','thanks','yes'])

# Load the saved model
model = load_model(r'C:\Users\KIIT\Desktop\project_video_chat\8thSemProject_part2\8thSemProject\action.h5')

sequence=[]
sentence = []
predictions = []
threshold = 0.7
count=31

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(70,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(70,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(70,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(70,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def PredictSign(frame):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Make detections
        global sequence
        global sentence
        global predictions
        global threshold
        global count

        # if len(predictions)>30:
        #     predictions.pop()
        # if len(sentence)>100:
        #     sentence.pop()

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence)>30:
            sequence= sequence[-30:]
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            if np.unique(predictions[-3:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            return (sentence[-1])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        return (sentence[-1])

            else:
                pass
        else:
            count-=1
            # return (f"Feeding model Please wait for {count} seconds")




# model = load_model('C:/Users/KIIT/Desktop/action.h5')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    return 'Running'

@socketio.on('connect')
def on_connect():
    emit('me', request.sid)

@socketio.on('disconnect')
def on_disconnect():
    send('callEnded', broadcast=True)

@socketio.on('predictionVideo')
def on_prediction_video(data):
    global actions
    img_bytes = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    ans=PredictSign(img)
    # Use the img for analysis or processing
    # if ans not in actions:
    #     ans='hi'
    print(ans)
    emit('predictionVideo', ans)


@socketio.on('callUser')
def on_call_user(data):
    from_user = data['from']
    userToCall = data['userToCall']
    caller_name = data['name']
    signal = data['signalData']
    emit('callUser', {'from': from_user, 'name': caller_name, 'signal': signal}, room=userToCall)


@socketio.on('answerCall')
def on_answer_call(data):
    emit('callAccepted', data['signal'],room=data['to'])

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)

