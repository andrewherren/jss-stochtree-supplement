# StochTree JSS Supplement

Supplementary replication materials for the stochtree JSS paper

## User Guide

### Prerequisites

The figures produced in the paper can be matched exactly using docker. 
To begin, first install [docker](https://docs.docker.com/desktop/).

Several of the demo datasets must be downloaded locally to the `data/` directory.
This can be done by running the python `scripts/data_download.py` script from the terminal. 
If you need to install python, there are many high-quality python distributions for different systems. For one example that works well on all platforms, we recommend [Anaconda](https://www.anaconda.com/download). Once a python environment is set up, download the data by running

```bash
python scripts/data_download.py
```

### Replicating R Output

First, build the R docker image via

```bash
docker build -t stochtree-r -f Dockerfile-R . 
```

To force a rebuild of the image, use the `--no-cache` option:

```bash
docker build --no-cache -t stochtree-r -f Dockerfile-R . 
```

and then run the replication script via

```bash
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/replication_script.R
```

### Replicating Python Output

First, build the python docker image via

```bash
docker build -t stochtree-python -f Dockerfile-Python . 
```

and then run the replication script via

```bash
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/replication_script.py
```
