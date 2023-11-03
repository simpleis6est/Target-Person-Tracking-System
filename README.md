# Title : Target-Person-Tracking-System
Target Person Tracking System

On this project, our goal is to integrate face recognition and person tracking, both of which are provided by Openvino. If the programme detects and identifies a specific individual, the software then checks whether the match rate is 70% or higher in relation to a previously trained target. Once this threshold is met (i.e., when the programme confirms the person to be recognized), the software draws a square around the person and continues to track them. Additionally, we utilized data augmentation to train the programme with images of the South Korean actor Wonbin. The integration is presented in merge.py.


### Virtual Enviroment
```sh
python3 -m venv target_tracking
```
### How to run
```
python ./face_recognition_demo.py 
-i 0 
-m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml 
-m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml 
-m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml 
--verbose 
-fg "/home/ubuntu/face_gallery"

```
### requirements.txt 

### AI model 1.facial recognition
- 

### AI model 2.Object Tracking
- Person Tracking with OpenVINO
