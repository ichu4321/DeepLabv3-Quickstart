# Steps to Install
	-- install Python 3.6
	-- create and activate a new virtual environment
	-- install the python requirements
		- For Linux: "pip install -r requirements_python36_ubuntu18.txt"
		- For Windows: "pip install -r requirements_python36_windows10.txt"

# Steps to Train
	-- Copy images into the TrainingImages folder
	-- copy labels into the TrainingLabels folder
	-- run "bash TRAIN.sh" (Linux) or "TRAIN.bat" (Windows)
		- creates "model.pb" after a few minutes
	-- run "python3 DEEPLAB_DEMO.py" to do a livecam test of the model
