import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
import vlc;
import time
from gtts import gTTS
import os
from datetime import datetime
import csv
import json

# ==================== CONFIGURATION ====================
class Config:
    # HIGHLY SENSITIVE Thresholds
    EYE_AR_THRESH = 0.15          # Higher = more sensitive
    EYE_CLOSED_FRAMES = 10       # ~0.33 seconds (very fast)
    YAWN_THRESH = 0.6
    YAWN_FRAMES = 10
    
    # Alert settings
    ALERT_COOLDOWN = 3            # Seconds between same alert type
    ENABLE_AUDIO = True
    ENABLE_LOGGING = True

# ==================== UTILITY FUNCTIONS ====================
def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))

def ear(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = euclideanDist(eye[1], eye[5])
    B = euclideanDist(eye[2], eye[4])
    C = euclideanDist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def yawn(mouth):
    """Calculate yawn ratio"""
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def calculate_perclos(ear_history, threshold=0.30):
    """Calculate PERCLOS (Percentage of Eye Closure)"""
    if len(ear_history) == 0:
        return 0
    closed_count = sum(1 for ear_val in ear_history if ear_val < threshold)
    return (closed_count / len(ear_history)) * 100

# ==================== DATA LOGGER ====================
class DataLogger:
    def __init__(self):
        self.session_start = datetime.now()
        self.log_file = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        self.events = []
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Event', 'EAR', 'PERCLOS', 'Alert_Count'])
    
    def log_event(self, event, ear, perclos, alert_count):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.events.append({
            'timestamp': timestamp,
            'event': event,
            'ear': ear,
            'perclos': perclos
        })
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event, f"{ear:.3f}", f"{perclos:.2f}", alert_count])
    
    def generate_report(self):
        """Generate session report"""
        duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        report = {
            'session_duration_mins': round(duration, 2),
            'total_events': len(self.events),
            'drowsy_events': sum(1 for e in self.events if 'DROWSY' in e['event']),
            'yawn_events': sum(1 for e in self.events if 'YAWN' in e['event'])
        }
        
        report_file = f"report_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

# ==================== STATISTICS TRACKER ====================
class StatsTracker:
    def __init__(self):
        self.ear_history = []
        self.yawn_count = 0
        self.alert_count = 0
        self.max_ear_history = 600  # 20 seconds at 30 fps
    
    def update_ear(self, ear_value):
        self.ear_history.append(ear_value)
        if len(self.ear_history) > self.max_ear_history:
            self.ear_history.pop(0)
    
    def get_perclos(self):
        return calculate_perclos(self.ear_history, Config.EYE_AR_THRESH)

# ==================== ALERT MANAGER ====================
class AlertManager:
    def __init__(self):
        self.last_alert_time = {}
        
        # Generate audio files
        self.generate_audio_files()
        
        # Load audio players
        self.alert_audio = vlc.MediaPlayer('drowsy_alert.mp3')
        self.break_audio = vlc.MediaPlayer('take_a_break.mp3')
        self.current_playing = None
    
    def generate_audio_files(self):
        """Generate all audio files"""
        audio_files = {
            'drowsy_alert.mp3': "Alert! Please take a break, drowsiness detected",
            'take_a_break.mp3': "Multiple alerts detected, you must take a break immediately"
        }
        
        for filename, text in audio_files.items():
            if not os.path.exists(filename):
                print(f"Generating {filename}...")
                tts = gTTS(text, lang='en')
                tts.save(filename)
    
    def trigger_alert(self, alert_type):
        """Trigger alert with cooldown management"""
        current_time = time.time()
        
        # Check cooldown
        if alert_type in self.last_alert_time:
            if (current_time - self.last_alert_time[alert_type]) < Config.ALERT_COOLDOWN:
                return False
        
        # Play appropriate audio
        if alert_type == "DROWSY":
            if self.current_playing != "DROWSY":
                self.alert_audio.stop()
                self.alert_audio.play()
                self.current_playing = "DROWSY"
        elif alert_type == "BREAK":
            if self.current_playing != "BREAK":
                self.break_audio.stop()
                self.break_audio.play()
                self.current_playing = "BREAK"
        
        self.last_alert_time[alert_type] = current_time
        return True
    
    def stop_all(self):
        """Stop all audio alerts"""
        self.alert_audio.stop()
        self.break_audio.stop()
        self.current_playing = None

# ==================== MAIN APPLICATION ====================
def main():
    print("=" * 70)
    print("        ADVANCED DROWSINESS DETECTION SYSTEM")
    print("=" * 70)
    print("  Status: INITIALIZING...")
    print("=" * 70)
    
    # Initialize components
    logger = DataLogger() if Config.ENABLE_LOGGING else None
    stats = StatsTracker()
    alert_manager = AlertManager()
    
    # Initialize video capture with HIGH QUALITY
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # HD width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # HD height
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    
    # Verify camera opened
    if not capture.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    print("  Status: ACTIVE - Monitoring started")
    print("  Controls: Press 'Q' to Exit | Press 'R' for Report")
    print("=" * 70 + "\n")
    
    # Counters
    eye_closed_counter = 0
    yawn_active = False
    frame_count = 0
    no_face_frames = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("WARNING: Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Process at full resolution for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in varying light
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with optimal parameters
        rects = detector(gray, 1)  # 1 = upsample once for better detection
        
        avgEAR = 0
        alert_status = "MONITORING"
        status_color = (0, 255, 0)
        
        if len(rects) > 0:
            no_face_frames = 0
            rect = rects[0]
            shape = face_utils.shape_to_np(predictor(gray, rect))
            
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            mouth = shape[mStart:mEnd]
            
            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0
            yawn_ratio = yawn(mouth)
            
            # Update statistics
            stats.update_ear(avgEAR)
            perclos = stats.get_perclos()
            
            # Draw face features
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            
            eye_color = (0, 255, 0)
            mouth_color = (0, 255, 0)
            
            # Yawn detection
            if yawn_ratio > Config.YAWN_THRESH:
                yawn_active = True
                stats.yawn_count += 1
                mouth_color = (0, 255, 255)
                cv2.putText(frame, "YAWN DETECTED!", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                yawn_active = False
            
            # Eye closure detection - VERY SENSITIVE
            if avgEAR < Config.EYE_AR_THRESH:
                eye_closed_counter += 1
                eye_color = (0, 255, 255)
                
                # Immediate alert for drowsiness
                if eye_closed_counter >= Config.EYE_CLOSED_FRAMES:
                    eye_color = (0, 0, 255)
                    alert_status = "DROWSINESS ALERT!"
                    status_color = (0, 0, 255)
                    
                    # Trigger alert
                    if alert_manager.trigger_alert("DROWSY"):
                        stats.alert_count += 1
                        if logger:
                            logger.log_event("DROWSY", avgEAR, perclos, stats.alert_count)
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Extra alert for yawn + drowsy
                    if yawn_active:
                        cv2.putText(frame, "CRITICAL: YAWN + DROWSY", (20, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (147, 20, 255), 3)
            else:
                # Reset when eyes open
                if eye_closed_counter > 0:
                    eye_closed_counter = 0
                    alert_manager.stop_all()
            
            # Draw contours
            cv2.drawContours(frame, [leftEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [rightEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [mouthHull], -1, mouth_color, 2)
            
            # Simple metrics display
            cv2.putText(frame, f"EAR: {avgEAR:.3f}", (20, frame.shape[0] - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Break reminder
            if stats.alert_count > 0 and stats.alert_count % 3 == 0:
                alert_manager.trigger_alert("BREAK")
                cv2.putText(frame, "TAKE A BREAK NOW!", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        else:
            no_face_frames += 1
            
            # Only show no face warning after brief period
            if no_face_frames > 10:
                cv2.putText(frame, "No Face Detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, "Please face the camera", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # If no face for long time, reset counters
                if no_face_frames > 90:  # 3 seconds
                    eye_closed_counter = 0
                    alert_manager.stop_all()
        
        # Clean status bar
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (40, 40, 40), -1)
        cv2.putText(frame, f"{alert_status} | Alerts: {stats.alert_count} | Yawns: {stats.yawn_count}", 
                   (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.imshow('Drowsiness Detection System', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            if logger:
                report = logger.generate_report()
                print("\n" + "=" * 50)
                print("SESSION REPORT")
                print("=" * 50)
                print(f"Duration: {report['session_duration_mins']} minutes")
                print(f"Total Events: {report['total_events']}")
                print(f"Drowsy Events: {report['drowsy_events']}")
                print(f"Yawn Events: {report['yawn_events']}")
                print("=" * 50 + "\n")
        
        if cv2.getWindowProperty('Drowsiness Detection System', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Cleanup and final report
    if logger:
        report = logger.generate_report()
        print("\n" + "=" * 70)
        print("FINAL SESSION REPORT")
        print("=" * 70)
        print(f"Session Duration: {report['session_duration_mins']} minutes")
        print(f"Total Events: {report['total_events']}")
        print(f"Drowsiness Alerts: {report['drowsy_events']}")
        print(f"Yawn Detections: {report['yawn_events']}")
        print(f"Data saved to: {logger.log_file}")
        print("=" * 70)
    
    alert_manager.stop_all()
    capture.release()
    cv2.destroyAllWindows()
    print("\nSystem stopped successfully!")

if __name__ == "__main__":
    main()

