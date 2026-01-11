from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import json

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureCounter:
    def __init__(self):
        self.count = 0  
        self.session_active = False  
        self.session_start_time = None  
        self.session_duration = 60  
        
        self.previous_y = None  
        self.movement_threshold = 0.05  
        self.direction = None  
        self.last_direction_change = time.time()  
        self.debounce_time = 0.3  
        self.position_buffer = deque(maxlen=5)
        self.best_score = 0  
        self.last_score = 0  

    def reset_session(self):
        self.count = 0
        self.previous_y = None
        self.direction = None
        self.position_buffer.clear()
        self.session_start_time = time.time()
        self.session_active = True

    def get_remaining_time(self):
        if not self.session_active or self.session_start_time is None:
            return self.session_duration
        elapsed = time.time() - self.session_start_time
        remaining = max(0, self.session_duration - elapsed)
        return remaining

    def is_session_complete(self):
        if not self.session_active:
            return False
        return self.get_remaining_time() <= 0

    def get_smoothed_position(self, y_position):
        self.position_buffer.append(y_position)
        return np.mean(self.position_buffer)

    def detect_gesture(self, hand_landmarks):
        
        if not self.session_active:
            return False

        wrist = hand_landmarks.landmark[0]
        current_y = wrist.y

       
        smoothed_y = self.get_smoothed_position(current_y)

        # Need at least 2 positions to detect movement
        if self.previous_y is None:
            self.previous_y = smoothed_y
            return False

        # Calculate vertical displacement
        displacement = smoothed_y - self.previous_y

        # Check if movement exceeds threshold
        if abs(displacement) > self.movement_threshold:
            current_time = time.time()
            
            # Debouncing
            if current_time - self.last_direction_change < self.debounce_time:
                return False

            
            new_direction = 'down' if displacement > 0 else 'up'
            
    
            gesture_completed = False
            if self.direction == 'up' and new_direction == 'down':
                self.count += 1
                gesture_completed = True

            
            self.direction = new_direction
            self.last_direction_change = current_time
            self.previous_y = smoothed_y
            
            return gesture_completed

        return False

    def calculate_rank(self):
        score = self.count
        if score >= 85:
            rank = "GOD"
            percentile = 95
        elif score >= 61:
            rank = "TOP TIER"
            percentile = 75
        elif score >= 31:
            rank = "MID TIER"
            percentile = 50
        else:
            rank = "NOOB"
            percentile = 25
            
        return rank, percentile, score


counter = GestureCounter()


camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 45)
    return camera

def generate_frames():
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1  
    ) as hands:
        
        camera = get_camera()
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            results = hands.process(rgb_frame)

            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    
                    if counter.session_active:
                        counter.detect_gesture(hand_landmarks)

            
            if counter.is_session_complete():
                counter.session_active = False
                counter.last_score = counter.count
                if counter.count > counter.best_score:
                    counter.best_score = counter.count

            
            remaining = int(counter.get_remaining_time())
            
            
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            
            
            cv2.putText(frame, f'Count: {counter.count}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Time: {remaining}s', (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            status = "ACTIVE" if counter.session_active else "READY"
            color = (0, 255, 0) if counter.session_active else (100, 100, 100)
            cv2.putText(frame, status, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route.
    Returns a multipart response with continuous JPEG frames.
    """
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_session', methods=['POST'])
def start_session():
    """API endpoint to start a new counting session"""
    counter.reset_session()
    return jsonify({'status': 'started', 'duration': counter.session_duration})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """
    API endpoint to get current session statistics.
    Returns count, time remaining, and rank information.
    """
    rank, percentile, score = counter.calculate_rank()
    
    return jsonify({
        'count': counter.count,
        'remaining_time': int(counter.get_remaining_time()),
        'session_active': counter.session_active,
        'rank': rank,
        'percentile': percentile,
        'score': score,
        'best_score': counter.best_score,
        'last_score': counter.last_score
    })

if __name__ == '__main__':
    # Run Flask development server
    # host='0.0.0.0' makes it accessible from other devices on network
    # debug=True enables auto-reload on code changes
    app.run(debug=True, host='0.0.0.0', port=5000)