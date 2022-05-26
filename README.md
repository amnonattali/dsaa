# Discrete State-Action Abstraction (DSAA)

## Files:
- dsaa.py
- baseline.py
- torch_models.py
- update_models.py
- utils.py
- vis.py
- experiments
    - easy_task
        - config.json
    - hard_task
        - config.json
- environments
    - dsaa_envs
    - setup.py
    - env_wrappers.py
- saved_data
    - saved returns from paper results
    - some examples of the output from vis.py

## Instructions for use:
- install necessary packages (tested with python 3.6)
    - For running experiments: torch (CPU), gym, numpy, pickle
    - For visualization: networkx, matplotlib
- register dsaa gym packages
    ```console
    $cd environments
    $pip install -e .
    ```
- run experiments (we provide the Arm2D experiments from Figure 4d in the paper)
    ```console
    $python dsaa.py --exp_path=experiments/[easy, hard]_task
    ```
- visualize various results
    ```console
    $python vis.py --exp_path=experiments/[easy, hard]_task [--make_cspace_vid] [--draw_abstract_mdp] [--make_sr_vis] [--paper_returns]
    ```