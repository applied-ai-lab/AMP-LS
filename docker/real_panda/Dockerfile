# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04
ENV LANG C.UTF-8
ENV TZS=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==================================================================
# User setting
# ------------------------------------------------------------------

ARG USER
ARG PASSWD
ARG USER_ID

RUN apt-get update && \
    apt-get install -y sudo

ENV HOME /home/$USER

RUN groupadd -g $USER_ID -o $USER && \
    useradd -m $USER -u $USER_ID -g $USER_ID && \
    gpasswd -a $USER sudo && \
    echo "$USER:$PASSWD" | chpasswd && \
    echo 'Defaults visiblepw' >> /etc/sudoers && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER
WORKDIR $HOME
ENV HOME $HOME

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    sudo rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    sudo apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        zsh \
        neovim \
        tmux \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    sudo make -j"$(nproc)" install

# ==================================================================
# zsh
# ------------------------------------------------------------------
RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    zsh && \
    sudo chsh -s $(which zsh) && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3-pip \
        python3.6 \
        python3.6-dev \
	python3-dev \
        python3-distutils-extra \
        ffmpeg \
        libsm6 \
        libxext6 \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    sudo ln -s /usr/bin/python3 /usr/local/bin/python && \
    pip3 install --upgrade pip

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    #PIP_INSTALL="pip install" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        setuptools && \
    $PIP_INSTALL \
        numpy \
        scipy \
        cloudpickle \
        Cython \
        tqdm \
        h5py \
        enum34 \
        pyyaml \
        wandb  \
        sympy  \
        scipy \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        opencv_python \
        trimesh \
        autolab_core \
        python-fcl \
        urdfpy \
        tensorboardX \
        funcsigs && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        freeglut3-dev \
        ninja-build \
        htop \
        swig \
        openssh-server \
        libeigen3-dev \
        liborocos-kdl-dev \
        libkdl-parser-dev \
        liburdfdom-dev \
        libnlopt-dev \
        libnlopt-cxx-dev \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libxext6 \
        libx11-6

# ==================================================================
# pytorch
# ------------------------------------------------------------------
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
    torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
    torch_scatter -f https://data.pyg.org/whl/torch-1.8.0+cu111.html

# ==================================================================
# zsh
# ------------------------------------------------------------------

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    #PIP_INSTALL="pip install" && \
    GIT_CLONE="git clone --depth 10" && \
    sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    $GIT_CLONE https://github.com/junjungoal/dotfiles.git && \
    sh dotfiles/dotfilesLink.sh && \
    $GIT_CLONE https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions && \
    $(which zsh) -c "source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" && \
    $(which zsh) -c "source ~/.zshrc" && \
    rm -rf ~/.vim && \
    $GIT_CLONE https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim && \
    mkdir .config && \
    cp -r dotfiles/config/* .config/


# ==================================================================
# ROS
# ------------------------------------------------------------------

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN sudo apt-get update

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    #PIP_INSTALL="pip install" && \
    GIT_CLONE="git clone --depth 10" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        ros-noetic-catkin \
        ros-noetic-desktop-full \
        ros-noetic-rviz \
        ros-noetic-rqt \
        ros-noetic-rqt-common-plugins \
        ros-noetic-franka-ros \
        python3-catkin-tools \
        python3-wstool \
        python3-rosdep

RUN mkdir -p ~/catkin_ws/src
RUN ["/bin/zsh", "-c", \
    "source /opt/ros/noetic/setup.zsh && \
    cd ~/catkin_ws && \
    catkin_make"]

RUN \
    sudo rosdep init && \
    rosdep update

# download moveit sources
RUN cd ~/catkin_ws/src \
    && wstool init . \
    && wstool merge -t . https://raw.githubusercontent.com/ros-planning/moveit/master/moveit.rosinstall \
    && wstool remove  moveit_tutorials  # this is cloned in the next section \
    && wstool update -t .

# download example code
RUN cd ~/catkin_ws/src \
    && git clone https://github.com/ros-planning/moveit_tutorials.git -b master \
    && git clone https://github.com/junjungoal/panda_moveit_config.git

RUN ["/bin/zsh", "-c", \
    "source /opt/ros/noetic/setup.zsh && \
    cd ~/catkin_ws && \
    rosdep install -y --from-paths src --ignore-src --rosdistro noetic"]

# # Download and build libfranka
# RUN ["/bin/zsh", "-c", \
#     "git clone --recursive https://github.com/frankaemika/libfranka $HOME/libfranka && \
#     mkdir $HOME/libfranka/build && \
#     cd $HOME/libfranka/build && \
#     cmake -DCMAKE_BUILD_TYPE=Release .. && \
#     cmake --build ."]


ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV LIBGL_ALWAYS_INDIRECT 0

RUN ["/bin/zsh", "-c", \
 "nvim +VundleInstall +qall"]
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN mkdir -p ~/projects

RUN sudo ldconfig && \
    sudo apt-get clean && \
    sudo apt-get autoremove && \
    sudo rm -rf /var/lib/apt/lists/* /tmp/*




