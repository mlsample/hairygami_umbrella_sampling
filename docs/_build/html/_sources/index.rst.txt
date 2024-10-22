.. _ipy_oxdna documentation:

===============================
Welcome to hairygami_umbrella_sampling's Documentation
===============================

 .. toctree::
    :maxdepth: 2
    :caption: Table of Contents

    Introduction <self>
    modules


Introduction
------------


`hairygami_umbrella_sampling` is a Python interface for running oxDNA umbrella sampling and large throughput simulations. This code is complementary to the article, if you use this code a citation would be appreciated:

- Sample, M., Liu, H., Diep, T., Matthies, M., & Šulc, P. (2023). Hairygami: Analysis of DNA Nanostructures’ Conformational Change Driven by Functionalizable Overhangs. ACS nano.

For more tutorials and examples, please refer to the `src` folder within this repository.

NVIDIA Multiprocessing Service (MPS)
------------------------------------
 .. code-block:: bash

   export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe
   export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY
   mkdir -p $CUDA_MPS_LOG_DIRECTORY
   nvidia-cuda-mps-control -d

NVIDIA MPS enhances the multiprocessing capabilities of CUDA-enabled GPUs

Prerequisites
-------------

- Docker image available on `Dockerhub <https://hub.docker.com/repository/docker/mlsample/hairygamiumbrellasampling/tags>`_.
 .. code-block:: bash
   
   docker pull mlsample/hairygamiumbrellasampling:latest
   docker run --rm --gpus all -it -p 8888:8888/tcp hairygamiumbrellasampling:latest

or

- Create a virtual python env, Python >= 3.10
- oxDNA installed with Python bindings.
- run 'pip install .' within the root directory of this repository.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
