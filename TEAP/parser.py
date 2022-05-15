####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import argparse

def str2bool(v):
    return v.lower() in ['t', 'true', True]

def str2list(v):
    if isinstance(v, list):
        return v
    else:
        return [item for item in v.split(',')]


def get_parser():
    parser = argparse.ArgumentParser()
    # General Settings
    parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')
    parser.add_argument('--seed', type=int, default=3, help='set seed')
    parser.add_argument('--mode', type=str, default='meta-test', help='meta-train|meta-test|nas')
    parser.add_argument('--main_path', type=str, default='.')
    parser.add_argument('--img_size', type=int, default=32, help='32|224')
    parser.add_argument('--metrics', type=str2list, default=["spearman"], help="metric for ranking correlation between real and estimated latencies of architectures.")
    parser.add_argument('--search_space', type=str, default='transnasbench', help='fbnet|nasbench201|ofa|transnasbench')
    parser.add_argument('--load_path', type=str, default='./data/transnasbench/checkpoint/help_max_corr.pt', help='model checkpoint path')    
    # Data & Meta-learning Settings
    parser.add_argument('--meta_train_devices', type=str2list, 
                default='autoencoder,class_object,class_scene,normal,room_layout')
    parser.add_argument('--meta_valid_devices', type=str2list, 
                default='segmentsemantic')
    parser.add_argument('--meta_test_devices', type=str2list, 
                default='jigsaw')
    parser.add_argument('--num_inner_tasks', type=int, default=5, help="the number of meta-batch")
    parser.add_argument('--num_meta_train_sample', type=int, default=900, help="the number of samples for each device in meta-training pool")
    parser.add_argument('--num_samples', type=int, default=10, help="the number of training samples for each task")
    parser.add_argument('--num_query', type=int, default=1000, help="the number of test samples for each task")
    parser.add_argument('--meta_lr', type=float, default=1e-4, help="meta-learning rate")
    parser.add_argument('--num_episodes', type=int, default=4000, help="the number of episodes during meta-training")
    parser.add_argument('--num_train_updates', type=int, default=2, help="the number of inner gradient step during meta-training")
    parser.add_argument('--num_eval_updates', type=int, default=2, help="the number of inner gradient step during meta-test")
    parser.add_argument('--alpha_on', type=str2bool, default=True, help="True:Ours&Meta-SGD/False:MAML")
    parser.add_argument('--inner_lr', type=float, default=1e-3, help="inner learning rate for MAML")
    parser.add_argument('--second_order', type=str2bool, default=True, help="on/off computing second order gradient of bilevel optimization framework (MAML framework)")
    # Save / Log
    parser.add_argument('--save_path', type=str, default='results', help='')
    parser.add_argument('--save_summary_steps', type=int, default=50, help="the interval to print log")
    # Wandb
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='help')
    parser.add_argument('--group', type=str, default=None)
    # Meta-learner
    parser.add_argument('--hw_embed_on', type=str2bool, default=True, help="on/off hardware-condtioned prediction")
    parser.add_argument('--hw_embed_dim', type=int, default=10, help="the dimension of hardware embedding")
    parser.add_argument('--layer_size', type=int, default=100, help="the size of hidden layer of the predictor")
    # Inference Network
    parser.add_argument('--z_on', type=str2bool, default=True, help="on/off z modulator")
    parser.add_argument('--determ', type=str2bool, default=False)
    parser.add_argument('--kl_scaling', type=float, default=0.1)
    parser.add_argument('--z_scaling', type=float, default=0.01)
    parser.add_argument('--mc_sampling', type=int, default=10)
    # Latency-constrainted NAS
    parser.add_argument('--sampled_arch_path', type=str, default='data/nasbench201/arch_generated_by_metad2a.txt', help="")
    parser.add_argument('--nas_target_device', type=str, default=None, help="target device of NAS process")
    parser.add_argument('--latency_constraint', type=float, default=None, help="latency constraint when performing NAS process")

    args = parser.parse_args()

    return args
