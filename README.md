# Deepfake-detection-of-image-audio-video
This project builds a multi-modal deepfake detection system using CNNs for images, MFCC with deep learning for audio, and InceptionV3-GRU for video analysis. It identifies facial inconsistencies, synthetic speech, and lip-sync issues, enhancing detection accuracy across media formats to combat misinformation and ensure digital content authenticity.
Deepfake Detection: Image, Audio & Video

A Streamlit web app that checks whether an image, audio clip, or video is real or a deepfake, with a separate deep learning model for each media type.

##Features
Media	How it works	Model file
Image	Resizes the image to 128×128 and classifies it as Real / Fake with a 5-layer CNN (PyTorch)	MODELS/cnn5.pth
Audio	Extracts 40 MFCC features (16 kHz, padded/trimmed to 500 frames) and classifies them with a Keras model; shows a confidence score	MODELS/updated_model.h5
Video	Samples 20 frames, crops the face in each (face_recognition), extracts features with InceptionV3 (ImageNet weights), then classifies the frame sequence with a Keras sequence model	MODELS/inceptionNet_model.h5

Supported uploads: jpg / jpeg / png, wav / mp3 / ogg, mp4 / avi / mov.

##Tech Stack
Python: PyTorch, torchvision, TensorFlow / Keras
Media processing: OpenCV, librosa, Pillow, face_recognition
UI: Streamlit
Project Structure
Deepfake-Detection-Extension/
├── DEEPFAKE_DETECTION_APP/
│   ├── app.py                    # Streamlit app (image, audio, video detection)
│   └── MODELS/
│       ├── cnn5.pth              # Image CNN weights (PyTorch)
│       ├── updated_model.h5      # Audio model (Keras)
│       ├── updated_model.keras   # Audio model (Keras native format)
│       └── inceptionNet_model.h5 # Video sequence model (Keras)
├── requirements.txt
└── README.md
Getting Started
bash
# 1. Clone the repo
git clone https://github.com/suyshx31/Deepfake-Detection-Extension.git
cd Deepfake-Detection-Extension

# 2. Create a virtual environment(Optional)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
cd DEEPFAKE_DETECTION_APP
streamlit run app.py

Then open the local URL Streamlit prints (usually http://localhost:8501).

Note: face_recognition depends on dlib, which needs CMake and a C++ compiler to install. See the face_recognition installation guide if pip install fails.

##Usage
Choose a media type: Image, Audio, or Video.
Upload a file.
See the result:
Image: Real / Fake
Audio: Real / Deepfake with a confidence percentage
Video: click Predict Deepfake to get REAL / FAKE
Results
<!-- Add your real numbers here: dataset used, accuracy / precision / recall per model. Only add figures you actually measured. -->
Model	Dataset	Accuracy
Image CNN	to add	to add
Audio (MFCC)	to add	to add
Video (InceptionV3 + sequence model)	to add	to add
Screenshots
<!-- Add 1–3 screenshots or a GIF of the app, e.g. ![App demo](assets/demo.png) -->
##Limitations
Video detection needs a visible face. Frames without one are skipped.
Each model only knows the kinds of fakes it was trained on, so results on new deepfake methods may be less reliable.
This is a research/academic project, not a forensic tool.
Future Work
Package it as a browser extension so you can scan media directly on web pages
Add audio-visual (lip-sync) consistency checks for videos
Deploy a public demo (e.g., Streamlit Community Cloud or Hugging Face Spaces)
Author

Suyash · GitHub @suyshx31
