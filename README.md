# Leveraging Scene Embeddings for Gradient-Based Motion Planning in Latent Space

[[Project website](https://amp-ls.github.io/)] [[Paper]()]

This project is a PyTorch implementation of Leveraging Scene Embeddings for Gradient-Based Motion Planning in Latent Space.

<p align="center">
    <img src="docs/video/real_static_furniture.gif" width=240 heigh=200>
    <img src="docs/video/real_dynamic.gif" width=240 heigh=200>
</p>

Motion planning framed as optimisation in structured latent spaces has recently emerged as competitive with traditional methods in terms of planning success while significantly outperforming them in terms of computational speed. However, the real-world applicability of recent work in this domain remains limited by the need to express obstacle information directly in state-space, involving simple geometric primitives. In this work we address this challenge by leveraging learned scene embeddings together with a generative model of the robot manipulator to drive the optimisation process. In addition we introduce an approach for efficient collision checking which directly regularises the optimisation undertaken for planning. Using simulated as well as real-world experiments, we demonstrate that our approach, AMP-LS, is able to successfully plan in novel, complex scenes while outperforming competitive traditional baselines in terms of computation speed by an order of magnitude. We show that the resulting system is fast enough to enable closed-loop planning in real-world dynamic scenes.


## Requirements

- python 3.7+

## Installation Instructions
# Clone the repository
```
git clone git@github.com:junjungoal/AMP-LS.git
```

Set the environment variable that specifies the root experiment directory. For example: 
```
mkdir ./experiments
export EXP_DIR=./experiments
```

## Docker Instructions
```
docker build -t lsmp:latest . --build-arg USER=username --build-arg PASSWD=hogehoge --build-arg USER_ID=$(id -u)
docker run -it -d -e DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw -v /home/username/projects/:/home/username/projects/ --net=host -e QT_X11_NO_MITSHM=1 -e XAUTHORITY=$XAUTH --privileged -v /dev:/dev  --gpus all -e HOST_UID=${UID} -e USER=username --name=lsmp  lsmp zsh
docker exec -it zsh
```

## Example Commands
To train a kinematics VAE model, run:
```
python -m core.train --path core/configs/model/kinematics_vae/panda --gpu 0 --prefix xxxx  --skip_first_val 1 --log_interval 100
```
To train a collision predictor with the pretrained kinematics VAE, run:
```
python -m core.train --path core/configs/model/constrained_kinematics_vae/panda --gpu 0 --prefix xxx  --skip_first_val 1 --log_interval 20
```

Results can be visualized using tensorboard in the experiment directory.

For planning in latent space with the learned VAE and collision predictor, run:
```
python -m core.rl.train --path core/configs/rl/lsmp/gazebo/panda --gpu 0 --prefix xxx  --mode traj_opt --n_val_samples 1
```
Make sure that you set a directory for the pre-trained VAE weight in `core/configs/rl/lsmp/gazebo/panda/conf.py`.

Results will be written to WandB.


## References
Codebase: https://github.com/clvrai/spirl 
