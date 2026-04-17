# ROO'YA (رؤية) - Smart Visual Assistance System 👁️

**ROO'YA** is an innovative AI-powered system designed to empower visually impaired individuals to navigate complex urban environments. It serves as a real-time "Visual Intelligence" dashboard, converting live visual data into clear, actionable audio cues.

---

##  Project Demonstration
The video below shows the **ROO'YA Interface** in action, demonstrating real-time detection and the automated voice alert system:

![Project Demo](رابط_الفيديو_هنا)

---

##  Project Overview
The **ROO'YA** project bridges the gap between AI technology and daily accessibility. The system is engineered to:
* **Identify Hazards:** High-speed detection of pedestrians, vehicles, and common street obstacles.
* **Directional Alerts:** Provides spatial audio feedback (Left, Right, Ahead) to help users orient themselves.
* **Interactive Monitoring:** A dedicated dashboard for developers and testers to monitor system performance, latency, and detection logs.
* **Smart Filtering:** Includes a cooldown mechanism to ensure that voice alerts remain clear and useful without overlapping.

---

##  Technical Stack
* **AI Core:** Custom-trained YOLO model (Optimized for urban detection).
* **Frontend:** Interactive Web Dashboard (HTML5, CSS3, JS).
* **Backend:** Python / Flask.
* **Audio:** Offline Text-to-Speech (TTS) for instant response.

---

##  How to Run
1. **Clone & Install:**
   ```bash
   git clone [https://github.com/yourusername/Rooya_Project.git](https://github.com/yourusername/Rooya_Project.git)
   pip install ultralytics flask opencv-python flask-cors pyttsx3
---

## ⚠️ IMPORTANT: Local Deployment
> [!IMPORTANT]
> **Privacy & Speed:** This system runs **ENTIRELY LOCAL**. It does not require an internet connection to process the camera feed or generate voice alerts. This ensures:
> - **Zero Latency:** Immediate response essential for safety.
> - **Privacy:** No data or video feed is ever uploaded to the cloud.
