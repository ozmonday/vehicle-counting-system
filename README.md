# Vehicle Counting System
This is a research project that has a goal for creating an application that can be used to count vehicle flow in roads . This project used YOLO algorithm for detecting vehicles and SORT algorithm for treacking objects.

**Accuration Model:**


**How to use:**
1. Download the tflite model from source https://drive.google.com/drive/folders/1QAH53oghpA64ZugPK4hGR1ApMEAw7UvR?usp=sharing
2. Create vitrual enivironment  `python3 -m virtualenv env`
3. Get all dependencies `pip install -r  requirments.txt`
4. Use command `python video_.py <video source> <virtual zone source> [<export path> <tflite model path> <classname model path>]`

**Download video and vitrual zone sample:** 
https://drive.google.com/drive/folders/16oDzZlZZXfNL3yDOkunn9EuRReMm
