docker run -it -d -v /home/${USER}/projects/:/home/${USER}/projects/ -v /home/${USER}/.ssh/:/home/${USER}/.ssh/ --net=host   --gpus all -e HOST_UID=${UID} -e USER=${USER} --name=lsmp lsmp
