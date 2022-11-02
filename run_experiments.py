from transfer_experiments.test_autoencoder import autoencoder_main
from transfer_experiments.transfer_experiments import transfer_exp
from transfer_experiments.transfer_utils import process_transfer_results
from high_dim_experiments.test_3joint_arm2d_img import arm2d_img
from high_dim_experiments.test_fourrooms_img_dsaa import fourrooms_img_dsaa
from high_dim_experiments.test_many_joint_arm2d import arm2d_many_joints
from experiments.baseline import train_baseline_policy
from experiments.dsaa import run_10_exps

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--experiment", dest="experiment", default="None")
    # dsaa, random, constrative, eigenoptions
    parser.add_argument("--transfer_exp_type", dest="transfer_exp_type", default="dsaa")
    
    # dsaa_experiments
    parser.add_argument("--exp_path", dest="exp_path", default="experiments/exp_test")
    parser.add_argument('--load_save', dest='load_save', action='store_true', default=False)

    args = parser.parse_args()

    if args.experiment == "autoencoder":
        autoencoder_main()
    elif args.experiment == "transfer_exp":
        transfer_exp(exp_type=args.transfer_exp_type)
    elif args.experiment == "process_transfer_results":
        process_transfer_results()
    elif args.experiment == "arm2d_img":
        arm2d_img()
    elif args.experiment == "fourrooms_img_dsaa":
        fourrooms_img_dsaa()
    elif args.experiment == "arm2d_many_joints":
        arm2d_many_joints()
    elif args.experiment == "train_baseline_policy":
        train_baseline_policy()
    elif args.experiment == "dsaa_experiments":
        import json

        with open("{}/config.json".format(args.exp_path)) as f_in:
            config = json.load(f_in)
            config["save_path"] = args.exp_path
            config["load_saved_abstraction"] = args.load_save
            
            run_10_exps(config)
    else:
        print("Need to specify a valid experiment to run")
