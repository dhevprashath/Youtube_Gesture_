YouTube Gesture Controller 🎬🤚

A gesture-controlled application to control YouTube playback and browser tabs using hand gestures and facial cues, leveraging computer vision. Built with Python (MediaPipe, OpenCV) and web automation, this project lets you navigate YouTube purely with gestures — no keyboard or mouse needed.

✅ What it does

Use hand gestures to control YouTube playback: play, pause, next video, previous video, volume control, etc.

Use additional gesture mappings (e.g. swipe, blink) to control browser tab switching, navigation, and video control.

Fully gesture-driven: no external hardware required — just a webcam.

Cross-platform (works on systems that support Python, OpenCV, MediaPipe, and a compatible browser).

📂 Project Structure (example — adapt based on your actual layout)



YouTube_Gesture/
├── main.py                 # Entry point — initializes webcam, gesture detection & controller logic
├── gesture_recognition.py  # Handles video capture and gesture detection (using MediaPipe / OpenCV)
├── controller.py           # Maps detected gestures to browser / YouTube actions (play, pause, next, etc.)
├── requirements.txt        # Python dependencies  
├── README.md               # Project documentation (this file)  
└── utils/                  # (Optional) Utility modules (e.g. logging, helper functions)


If your structure is different, please adjust accordingly.

🛠️ Prerequisites

Python 3.x

A webcam (built-in or external)

A web browser (e.g. Chrome, Firefox)

Python packages: OpenCV, MediaPipe (or equivalent), PyAutoGUI / selenium / browser automation tool (depending on implementation) — see requirements.txt or below.

📥 Installation & Setup

Clone the repository

git clone https://github.com/dhevprashath/YouTube_Gesture.git
cd YouTube_Gesture


(Optional) Create and activate a virtual environment

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Launch the application

python main.py
