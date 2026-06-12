# Fire-Front Propagation using the Level-Set Method

## Overview

This project reproduces a published NIST fire-front propagation benchmark using the Eulerian Level-Set Method.

## Features

- Eulerian Level-Set formulation
- Flux-limited upwind discretization
- SSP-RK2 time integration

## Tools Used

- Python
- NumPy
- Matplotlib
## Results

### Wind-Driven Circular Fire Front

![Figure 1](figure1.png)

### Level-Set vs MOL Comparison

![Figure 2](figure2.png)

### Front Merger

![Figure 3](figure3.png)

### Fuel Pocket Formation

![Figure 4](figure4.png)

### Scalloped Fire Front Formation

![Figure 5](figure5.png)

### Grid Convergence Study

![Convergence](convergence.png)

## Validation

- Reproduced the NIST fire-front propagation benchmark.
- Implemented flux-limited upwind discretization with SSP-RK2 time integration.
- Achieved less than 1% error relative to published benchmark data.

## Author

Siddhi Sharma  
Department of Aerospace Engineering  
Indian Institute of Technology Madras
