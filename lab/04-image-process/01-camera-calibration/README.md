# Run Pose #
## Step 1: Get Image ##
```bash
cd $project_folder
cd src
pipenv shell
python get-image.py
```

## Step 2: Calcuate the the camera matrix and rotation/transpose matrix
```bash
python calbration.py
```

## Step 3: Calcuate the pose
``` base
python pose.py
```
