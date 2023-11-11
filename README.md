# test_mediapipe - testing mediapipe pose esitmation
## bicep curl counter / body language detection with custom models 
## Quick Start

1. Clone the repo: `git clone <repo URL>`
2. Activate python virtualenv: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. To run the Bicep Curl Counter: run `python3 bicepCurlCounter.py`
5. To run the Body Language detector: run `python3 bodyLanguageDetector.py`

*** If the program crashes when trying to run, make sure python has access to your webcam and rerun
#### To close cv2 window: click q key
#
#

## To add new poses in the Body Language Detector:
1. rename `class_name` variable in `captureLandmarks.py`, Line 25 to new pose name
2. run `python3 captureLandmarks` and make sure the new pose data is added to the bottom of `coords.csv` 
3. train the model with the new pose by running `trainModel.py`
4. run the `python3 bodyLanguageDetector.py` script now and test your new pose



