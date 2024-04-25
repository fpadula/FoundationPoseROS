# FROM wenbowen123/foundationpose
FROM foundationpose_ros

# Setting up keys and repos
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# ROS
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-noetic-ros-base \
    python3-catkin-tools \
    ros-noetic-tf2-ros \
    ros-noetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

RUN pip install empy
RUN pip install catkin_tools
RUN pip install rospkg
RUN pip install hydra-core
RUN pip install hydra-colorlog
RUN pip install gdown
RUN pip install distinctipy
RUN pip install pytorch-lightning==1.8.1

ENV ROS_DISTRO=noetic