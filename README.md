# StochTree JSS Supplement

Supplementary replication materials for the stochtree JSS paper

## User Guide

### Prerequisites

The figures produced in the paper can be matched exactly using docker. 
To begin, first install [docker](https://docs.docker.com/desktop/).

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

For debugging purposes, you can run any of the analyses as a standalone script via

```bash
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/acic-bcf.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/custom-interface-bart-linreg.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/custom-interface-bart-robust.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/friedman-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/motorcycle-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/rdd-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/simulated-bcf.R
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

For debugging purposes, you can run any of the analyses as a standalone script via

```bash
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/acic-bcf.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/custom-interface-bart-linreg.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/custom-interface-bart-robust.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/friedman-bart.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/motorcycle-bart.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/rdd-bart.py
docker run --rm -v $(pwd):/workspace stochtree-python python scripts/Python/simulated-bcf.py
```
