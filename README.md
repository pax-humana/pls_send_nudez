# pls send nudez

Updated for Tensorflow2 and OpenNSFW2

## [Install and configure Tensorflow](https://www.tensorflow.org/install/pip#ubuntu_1804_cuda_101)

### Install Miniconda
`curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`

### Create a conda environment

`conda create --name opennsfw2 python=3.9`
`conda activate opennsfw2`

### GPU Setup

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`
`mkdir -p $CONDA_PREFIX/etc/conda/activate.d`
`echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`

### Install TensorFlow

`pip install --upgrade pip`
`pip install tensorflow`

## Running

Activate the conda environment you created above

`conda activate opennsfw2`

```
usage: pls_send_nudez.py [-h] [-s MIN_SIZE] [-7] output

positional arguments:
  output                Output folder name.

optional arguments:
  -h, --help            show this help message and exit
  -s MIN_SIZE, --min-size MIN_SIZE
                        Minimum image size in kilobytes. (Default: 20480)
  -7, --new-hash-length
                        Use image filename length of 7. (Default is 5)
```

```
usage: extract_nudez.py [-h] [-t THRESHOLD] [directory] [output]

positional arguments:
  directory             Input Directory. Default: current working directory.
  output                Output Subdirectory. Default: wins

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Default matching threshold for the open_nsfw model (0 - 1). Default: .7"
```

```
usage: score_nudez.py [-h] [directory]

positional arguments:
  directory   Input Directory. Default: current working directory.

optional arguments:
  -h, --help  show this help message and exit
```
