# FROM wenbowen123/foundationpose
FROM foundationpose_ros

RUN mv /etc/sudoers /etc/sudoers.bkp

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
    ros-noetic-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

RUN mv /etc/sudoers.bkp /etc/sudoers
ENV ROS_DISTRO=noetic