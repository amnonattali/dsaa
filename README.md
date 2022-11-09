# Discrete State-Action Abstraction (DSAA)

To quickly find the code that does most of the heavy lifting you should navigate to **update_models.py** in which we have two functions, one for learning a discrete abstraction based on a dataset of transitions, and another for training inter-state options based on a similar dataset and a learned abstraction. We provide comments for understanding the code along with notes about things we tried that didn't work.

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
- run the main transfer experiments (Figure 6 and Table 1 in the paper)
    ```console
    $python run_experiments.py --experiment=transfer_exp --transfer_exp_type=[dsaa, contrastive, eigenoptions, random, SRoptions]
    ```
- run experiments for Arm2D (Figure 7 in the paper)
    ```console
    $python run_experiments.py --experiment=dsaa_experiments --exp_path=experiments/[easy, hard]_task
    ```
- visualize various results
    ```console
    $python vis.py --exp_path=experiments/[easy, hard]_task [--make_cspace_vid] [--draw_abstract_mdp] [--make_sr_vis] [--paper_returns]
    ```