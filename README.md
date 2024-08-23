# act-plus-plus-ys
## Purpose of this repo

This repo is a version of
https://github.com/MarkFzp/act-plus-plus
that can be easily installed and ran on Windows (It worked on Windows 11 and 10). 
It was installed and tested in 03.2024. The code can be  also installed and executed on any other OS (I did not test it, but I assume it).

## Installation

step 1: clone this repo to your local machine.

step 2: consider that this project was 
        tested with python version 3.12.2.   

step 3:
install all modules from `requirements.txt` using pip or any other packet manager (conda etc.). 
Install to virtual environment or to global environment, whatever you prefer. 
Ideally, things should go smoothly and there should be no need for any further actions (building libraries etc.).
... not all modules are required for the project. So if you have
difficulties with installation of some module try running the project (see below) without it.


step 4:
open `detr` folder (from this project) in your terminal and run `pip install -e .`

## Generate demonstrations
To generate demonstrations run `ys_record_sim_episodes.py`
It is possible that you might need to adjust names of certain folders, for example name of folder where demonstrations will be stored. 
To visualize demonstration (to generate video), run `ys_episode_to_video.py`.

## Train the Model
To train the model, run `ys_imitate_episodes.py`
if you want to train on GPU, make sure that `gpu=True` (in this program).


## Visualize training stats
For the case that you are not using `wandb` for analyzing training results (`log_to_wandb=False` (in ys_imitate_episodes)):
Training stats such as validation set loss etc. will be logged to `log` directory.
Some programs for optional postprocessing are located in `postprocess_train` folder. 
Programs that visualize training statistics are located in `visualize_train` folder.

## rollout trained policy and create video
to rollout trained policy and create video, run `rollout_policy_to_video.py`

## Some tips concerning errors (when running the code)
tip 1:
When you encounter problems with `wandb`, and for the moment you don't want to bother about 
wandb account etc., simply make sure that `log_to_wandb=False` (in `ys_imitate_episodes.py`).


## And what now ?
Enjoy !
