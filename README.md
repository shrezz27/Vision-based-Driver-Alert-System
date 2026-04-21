# 🚗 Vision-based Driver Alert System
AI-powered vision-based system that detects driver drowsiness and triggers real-time alerts using facial landmark analysis
## 📌 Features

- 👁️ Eye Aspect Ratio (EAR) based drowsiness detection  
- 😮 Yawn detection using mouth aspect ratio  
- 📊 PERCLOS (Percentage of Eye Closure) monitoring  
- 🔊 Real-time audio alerts (Text-to-Speech)  
- 📁 Session logging (CSV + JSON reports)  
- ⚡ High sensitivity and fast response  
- 📷 Live webcam-based monitoring  

---

## 🧠 Tech Stack

- Python  
- OpenCV  
- Dlib  
- NumPy  
- VLC (Audio Alerts)  
- gTTS (Text-to-Speech)  

---

## 🏗️ Project Structure


├── main.py
├── main_dlib.py
├── final-integration.py
├── shape_predictor_68_face_landmarks.dat
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── requirements.txt


---

## ⚙️ How It Works

1. Captures real-time video from webcam  
2. Detects face and extracts facial landmarks  
3. Computes:
   - Eye Aspect Ratio (EAR)
   - Yawn Ratio
   - PERCLOS  
4. If thresholds are exceeded:
   - Triggers alert sound  
   - Logs event  
   - Displays warning on screen  

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/driver-alert-system.git
cd driver-alert-system
pip install -r requirements.txt
```
▶️ Usage
python final-integration.py

Controls:
Q → Quit
R → Generate report

📊 Output
Real-time EAR & PERCLOS values
Drowsiness alerts on screen
Audio warnings
CSV logs + JSON report

📈 Key Parameters
EAR Threshold: 0.15
Eye Closure Frames: 10
Yawn Threshold: 0.6

📂 Logs & Reports
CSV file for event logging
JSON file for session summary
Includes:
Duration
Total alerts
Drowsy events
Yawn events

⚠️ Limitations
Requires proper lighting
Works best for a single face
Needs webcam access
💡 Future Improvements
Mobile app integration
Night vision support
Deep learning model (CNN/LSTM)
Head pose estimation
Cloud-based analytics

🤝 Contributing
Pull requests are welcome. For major changes, open an issue first.

📜 License
MIT License
