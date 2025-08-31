#  Real-Time Hand Gesture Recognition  

This project implements a **real-time static hand gesture recognition system** using a standard webcam.  
It leverages a **rule-based approach** on top of a powerful hand-tracking library to accurately identify a vocabulary of **seven distinct gestures**.  
The application also includes **detection smoothing** to provide a stable and user-friendly experience.  

---

##  Recognized Gestures  

- 👍 **Thumbs Up**  
- ✊ **Fist**  
- ✌️ **Peace**  
- 👌 **OK**  
- 🖐️ **Open Palm**  
- 👆 **Pointing**  
- 🤞 **Crossed Fingers**  

---

##  Author  

**Your Full Name**  
Hasya Chaudhary  

---

##  Technology Justification  

The core technologies chosen for this project were **MediaPipe, OpenCV, and NumPy**. This combination provides a **robust, high-performance, and accessible framework** for solving real-time computer vision problems.  

### 🔹 MediaPipe  
- **State-of-the-Art Accuracy:** Provides **21 high-fidelity 3D landmarks** for each detected hand, eliminating the need to train a custom deep learning model.  
- **Exceptional Performance:** Optimized for **real-time inference** on standard CPU hardware.  
- **Robustness:** Pre-trained on a **diverse dataset**, works under different lighting conditions, hand sizes, and skin tones.  

### 🔹 OpenCV (Open Source Computer Vision Library)  
- **Camera Interfacing:** Uses `cv2.VideoCapture` to capture live video frames.  
- **Image Processing & Rendering:** Converts images to correct formats, draws landmarks, bounding boxes, and overlays gesture labels.  

### 🔹 NumPy  
- **Numerical Computations:** Converts landmark data into arrays for easy manipulation.  
- **Distance Calculation:** Used for gestures like **OK** and **Crossed Fingers** by computing Euclidean distances.  

---

##  Gesture Logic Explanation  

The system uses **geometric rules** based on the **21 landmarks from MediaPipe**.  
A helper function checks if fingers are **extended** or **curled**:  
- A finger is **extended** if its tip is higher than its middle joint.  

### ✨ Methodology for Key Gestures  

- **👍 Thumbs Up**  
  - Thumb tip above its joint.  
  - Other four fingers curled.  

- **✊ Fist**  
  - All fingers curled (including thumb).  

- **✌️ Peace**  
  - Index and middle fingers extended.  
  - Ring and pinky curled.  
  - Thumb state ignored.  

- **👌 OK**  
  - Distance between thumb tip & index tip calculated.  
  - Dynamic threshold (`0.18 * hand_diagonal`) ensures scaling works with hand distance from camera.  

---

## 🔄 Detection Smoothing  

To avoid **flickering outputs**, the app uses a **deque buffer** (size 6).  
- Only when one gesture is the **majority in buffer**, it gets displayed.  
- Ensures **stable and smooth recognition**.  

---

##  Setup and Execution Instructions  

### 1️⃣ Prerequisites  
- **Python 3.8+** installed.  
- A **webcam** connected.  

### 2. Clone or Download the Code
If using Git, clone the repository to your local machine:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

Alternatively, just download the main.py script and save it in a new folder.

### 3. Create a Virtual Environment  
```bash
python3 -m venv venv  
source venv/bin/activate
```

### 4. Install Dependencies  
```bash
pip install -r requirements.txt  
```

### 5. Run the Application  

Execute the script from your terminal:  

```bash
python main.py
```

 A window will pop up showing your webcam feed.  
 Position your hand in the frame to see the detected landmarks and the recognized gesture displayed in the top-left corner.  
 To quit the application, press the `ESC` key.  
