# Get the Camera Matrix and Estimate the Pose
## Command ##
```bash
cd $project_folder/src
pipenv shell
python main.py
```

### Option 1: Get Image ##
```
Input: None. 
Output: ./images/img?.png
```

## Option 2: Calcuate the the camera matrix and the rotation/transpose matrix
```
Input: None. 
Output: calibration.pkl
```

## Option 3: Calcuate the pose and project a cubic on the chessboard
```
Input calibration.pkl
Output: Cubic show
```

## Option 4: Calcuate the pose and project a cubic on the chessboard (camera version)
```
Input calibration.pkl
Output: Cubic show
```

## Option 5: Exit


# Reference
```bash 
https://github.com/niconielsen32/CameraCalibration/blob/main/calibration.py
```
