# Training
## Model training
To train the model, first download the dataset from Google Drive and extract it inside the directory `/tarmass/src/images/train`. \
Inside the docker container run:

```commandline
cd /tarmass/src/images/train
dvc pull dataset_100.zip.dvc
unzip dataset_100.zip
```

Next, go to the model directory and edit hyperparameters in the `config.py` configuration file if needed, then
run a training script:

```commandline
cd /tarmass/src/keypoints_detector
./train.py
```

Training visualizations are available on Tensorboard. \
Tensorboard results can be obtained in the directory `/tarmass/src/keypoints_detector/runs/board_results`. \
Losses are saved as an image file `/tarmass/src/keypoints_detector/training_results/training_results.png`. \
Model checkpoints are available in the directory `/tarmass/src/keypoints_detector/checkpoints`.

## Hyperparameter tuning
To find the optimal training hyperparameters, run the script:
```commandline
cd /tarmass/src
./hyperparam_search.py
```
One can edit the search space inside the `hyperparam_search.py` script.
