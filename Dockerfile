FROM ubuntu:18.04 as runner

WORKDIR "/build_space"

# Installs and downloads
RUN apt update && \
   apt install -y build-essential wget libssl-dev vim software-properties-common && \
   # Because the default version of git is too old to work with submodules
   add-apt-repository -y ppa:git-core/ppa && \
   apt install -y git && \
   # OpenGL-related libraries
   apt-get install -y libglu1-mesa-dev mesa-common-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev && \
   # CMake
   wget "https://github.com/Kitware/CMake/releases/download/v3.25.0-rc2/cmake-3.25.0-rc2.tar.gz" && \
   tar -xf cmake-3.25.0-rc2.tar.gz && \
   # Set up python for scripts
   apt install -y python3-pip zlib1g-dev libjpeg62 libjpeg62-dev && \
   python3 -m pip install numpy && \
   python3 -m pip install matplotlib && \
   python3 -m pip install boto3 && \
   python3 -m pip install dataclasses && \
   # Backup copies of what would normally be submodules
   git clone https://github.com/therealjtgill/glm && \
   git clone https://github.com/glfw/glfw

# Build CMake
WORKDIR "/build_space/cmake-3.25.0-rc2"
RUN ./bootstrap && \
   make -j4 && \
   make install

# Build glfw
WORKDIR "/build_space/glfw/build"
RUN cmake .. && \
   make -j4 && \
   make install

# Build glm
WORKDIR "/build_space/glm/build"
RUN cmake .. && \
   make -j4 && \
   make install

# -----------------------------------------------------

FROM intel/oneapi-vtune:latest as profiler

WORKDIR "/build_space"

# Installs and downloads
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
   apt install -y build-essential libssl-dev vim software-properties-common && \
   # Because the default version of git is too old to work with submodules
   add-apt-repository -y ppa:git-core/ppa && \
   apt install -y git && \
   # Python
   apt install -y python3-pip zlib1g-dev libjpeg62 libjpeg62-dev && \
   python3 -m pip install numpy && \
   python3 -m pip install matplotlib && \
   python3 -m pip install boto3 && \
   python3 -m pip install dataclasses && \
   # CMake
   wget "https://github.com/Kitware/CMake/releases/download/v3.25.0-rc2/cmake-3.25.0-rc2.tar.gz" && \
   tar -xf cmake-3.25.0-rc2.tar.gz

# Build CMake
WORKDIR "/build_space/cmake-3.25.0-rc2"
RUN ./bootstrap && \
   make -j4 && \
   make install

# -----------------------------------------------------

FROM nvidia/cuda:10.2-devel-ubuntu18.04 as cuda_runner

WORKDIR "/build_space"

RUN apt update && \
   apt install -y build-essential wget libssl-dev vim software-properties-common && \
   apt install -y software-properties-common && \
   # Because the default version of git is too old to work with submodules
   add-apt-repository -y ppa:git-core/ppa && \
   apt install -y git && \
   # OpenGL-related libraries
   apt-get install -y libglu1-mesa-dev mesa-common-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev && \
   # Set up python for scripts
   apt install -y python3-pip zlib1g-dev libjpeg62 libjpeg62-dev && \
   python3 -m pip install numpy && \
   python3 -m pip install matplotlib && \
   python3 -m pip install boto3 && \
   python3 -m pip install dataclasses && \
   # CMake
   wget "https://github.com/Kitware/CMake/releases/download/v3.25.0-rc2/cmake-3.25.0-rc2.tar.gz" && \
   tar -xf cmake-3.25.0-rc2.tar.gz

WORKDIR "/build_space/cmake-3.25.0-rc2"
RUN ./bootstrap && \
   make -j4 && \
   make install

# Run interactively:
#   docker run --gpus all -it [image name] /bin/bash

# Verify that it works:
#   docker run --gpus all -it [image name] nvidia-smi