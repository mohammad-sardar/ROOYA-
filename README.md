# Assistive Vision AI

Assistive Vision AI is a state-of-the-art application designed to assist visually impaired users by providing real-time object detection, hazard identification, and spoken alerts. The system utilizes advanced computer vision technology, powered by YOLOv8, to detect objects, estimate distances, and provide voice feedback to users about nearby hazards.

## User Interface & Control
![ROO'YA (رؤية) Dashboard](https://github.com/mohammad-sardar/ROOYA-/blob/main/Screenshot%202026-04-18%20030226.png)

### Key Features:
- **Live Detection Mode (LIVE Mode)**: The application continuously scans the user's environment in real-time, detecting potential obstacles or hazards using a live camera feed. It ensures that users receive constant, up-to-date hazard alerts.
  
- **Audio Feedback**: As part of its accessibility features, the system provides spoken alerts when a hazard is detected, helping users navigate safely without needing to look at the screen.

- **Neural Network Integration**: The app runs on a trained YOLOv8 model, which has been optimized for real-time object detection. This deep learning model ensures accurate identification and classification of obstacles, providing users with useful feedback.

### Application Interface Explanation:
- **Live Neural Vision**: Displays the live camera feed from the user's device, showing the detected objects and hazards in the environment. The detection highlights potential hazards with visual indicators.
  
- **Performance Monitor**: Shows real-time performance data, including:
  - **FPS (Frames Per Second)**: The current frame rate of the camera feed.
  - **Latency**: The delay in object detection and hazard alert processing.
  - **Detections**: The number of objects detected in the current session.
  - **Hazards**: The number of potential hazards identified in the current session.

- **Live Hazard Log**: Displays a list of detected hazards in real-time. It shows any detected hazards along with their classification, helping users track risks in the environment.

- **Current Mode**: Displays the current operational mode of the system, such as "Live Detection Active" or "System Standby".

- **Camera Controls**:
  - **Auto-detect Camera**: Automatically detects the camera connected to the device.
  - **Start Camera**: Begins the live feed, enabling real-time hazard monitoring.
  - **Stop Camera**: Stops the live camera feed.
  
- **Upload Test Video**: Allows users to upload pre-recorded videos for testing the model's performance with specific footage.

- **Mute Spoken Alert**: Enables or disables the spoken hazard alerts feature.

- **Training Hub**: Upload a custom YOLOv8 model checkpoint to fine-tune the system for specific use cases. This allows users to improve detection accuracy or customize the system for particular environments.

- **Detected Object**: Marks any object detected in the camera feed with a red indicator.
- **Hazard Under 2 Meters**: Indicates any hazard within a 2-meter range using a different color or icon for quick identification.
- **Glass Overlay Telemetry**: Adds a glass overlay visualization on the detected objects, helping users see the depth of objects in the environment.

### How to Use:
1. **Set Up the Camera**: Ensure that your camera is connected and detected by the system. You can use the "Auto-detect Camera" option for ease of use.
2. **Start the Detection**: Click on "Start Camera" to begin the real-time hazard monitoring.
3. **Receive Alerts**: As the system detects objects and hazards, you will receive both visual markers and spoken feedback.
4. **Upload a Test Video**: If you want to test the system with a pre-recorded video, you can use the "Upload Test Video" option.
5. **Train the Model**: If needed, upload your custom model through the "Training Hub" to tailor the system to specific environments or tasks.

### Watch the Application in Action
To see the application live and how it performs, click the link below to watch the demo video of the system working in real-time.

