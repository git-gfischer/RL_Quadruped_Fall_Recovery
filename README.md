# Quadruped Robot Fall Recovery Project
In this repo is implemented a fall recovery system using reinforcement learning as a final project for a deep learning course.

## Docker Installation
For easy installation of all dependecies of the project, run the following command. This might take some time depending on your internet connection
```
git clone https://github.com/git-gfischer/RL_Quadruped_Fall_Recovery.git
cd RL_Quadruped_Fall_Recovery
docker build -t isaaclab_dls .
```

## Enter the docker 
Once the docker image is built, enter the docker using docker compose
```
xhost +
docker compose -f docker/docker-compose.yaml run isaaclab bash
python3 -m pip install -e source/basic_locomotion_dls_isaaclab
```

## Train Policy
To train the policy for fall recovery, run the following command. It will ask for your wandb credentils. 
```
python3 scripts/rsl_rl/train.py --task=Recovery-Aliengo-Flat --num_envs=8000 --headless
```

## Play Policy
Once the policy is trained, you can use it running 
```
python3 scripts/rsl_rl/play.py --task=Recovery-Aliengo-Flat --num_envs=1
```