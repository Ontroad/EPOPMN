# EPOPMN

Code release for the paper: "Exterior Penalty Policy Optimization with Penalty Metric Network under Constraints", IJCAI 2024.

This repository is based on the [omnisafe](https://github.com/PKU-Alignment/omnisafe).

# Installation Guide

Create conda env: ```conda create -n epopmn python=3.8``` 

Activate conda env: ```conda activate epopmn```

Install: ```conda env create -f epopmn.yaml```

# Run Experiments

Register the EPO algorithm into the omnisafe library

Activate conda env: ```source activate epopmn```

Run the code by:
```
cd experiments
python train_exp.py --algo EPO_PMN --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
```

# License

EPOPMN is released under Apache License 2.0.
