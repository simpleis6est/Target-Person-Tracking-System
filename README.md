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
- The code snippet has code lines that obtain target individual's images from a directory. The software trains itself on the images for precision. 

### AI model 2.Object Tracking
- Person Tracking with OpenVINO
- Based on training results, this code snippet enables articulating the desirable individual when he/she is caught on camera by drawing a square around them. 

### Image augmentation
- For further precision, we added image augmentation to train the programme. 
