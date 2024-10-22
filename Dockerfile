# Use the specified base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    openssl \
    ca-certificates \
    git \
    cmake \
    make \
    g++ \
    vim \
    bash-completion \
    wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.12 python3-pip

# Install pip packages
RUN pip3 install \
    py \
    certifi \
    ipywidgets \
    pandas \
    nvidia-ml-py3 \
    statsmodels \
    sphinx \
    sphinx_rtd_theme \
    scienceplots \
    jupyterlab \
    tqdm

# Set the working directory to /root (home directory for the root user)
WORKDIR /root

# Clone and build oxDNA in /root directory
RUN git clone https://github.com/lorenzo-rovigatti/oxDNA.git && \
    cd oxDNA && \
    mkdir build && \
    cd build && \
    cmake ../ -DCUDA=1 -DPython=1 -DOxpySystemInstall=1 && \
    make -j 12 && \
    make install -j 12

# Clone hairygami_umbrella_sampling repo into /root directory
RUN git clone https://github.com/mlsample/hairygami_umbrella_sampling.git && \
    cd hairygami_umbrella_sampling && \
    pip install -e . && \
    chmod +x ./install_wham.sh && \
    ./install_wham.sh

# Customize bash environment for a nicer looking shell
RUN echo 'export PS1="\[\e[1;32m\]\u@\h:\[\e[1;34m\]\w\[\e[0m\]$ "' >> /root/.bashrc && \
    echo 'alias ll="ls -alF"' >> /root/.bashrc && \
    echo 'alias la="ls -A"' >> /root/.bashrc && \
    echo 'alias l="ls -CF"' >> /root/.bashrc && \
    echo 'alias h="history"' >> /root/.bashrc && \
    echo 'export CLICOLOR=1' >> /root/.bashrc && \
    echo 'export LSCOLORS=GxFxCxDxBxegedabagaced' >> /root/.bashrc && \
    echo 'source /etc/bash_completion' >> /root/.bashrc

# Set environment variables
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV LC_ALL=C
ENV SHELL="/bin/bash"

# Expose the port Jupyter will run on
EXPOSE 8888

# Set the default command to run when starting the container
CMD ["/bin/bash"]
