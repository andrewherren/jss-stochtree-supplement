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

and then run the replication scripts via

```bash
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/acic-bcf.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/custom-interface-bart-linreg.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/custom-interface-bart-robust.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/friedman-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/motorcycle-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/rdd-bart.R
docker run --rm -v $(pwd):/workspace stochtree-r Rscript scripts/R/simulated-bcf.R
```
