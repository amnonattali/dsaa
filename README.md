# Discrete State-Action Abstraction (DSAA)

## Files:

## Instructions for use:
- install necessary packages (tested with python 3.6)
    - For running experiments: torch (CPU), gym, numpy, pickle
        - NOTE: our gym environments are not compatible with the newest gym versions. Please use gym==0.18.3
    - For visualization: networkx, matplotlib
- register dsaa gym packages
    ```console
    $cd environments
    $pip install -e .
    ```
- run transfer experiments
    ```console
    $python run_experiments.py --experiment=transfer_exp --transfer_exp_type=[dsaa, contrastive, eigenoptions, random]
    ```
- run experiments (we provide the Arm2D experiments from Figure 7 in the paper)
    ```console
    $python run_experiments.py --experiment=dsaa_experiments --exp_path=experiments/[easy, hard]_task
    ```
- visualize various results
    ```console
    $python vis.py --exp_path=experiments/[easy, hard]_task [--make_cspace_vid] [--draw_abstract_mdp] [--make_sr_vis] [--paper_returns]
    ```