# ===== Schema & Argument Code Prompting Ablation Study.py ===== #

# Common Library
import os
import torch
import json
import argparse
from tqdm import tqdm
from transformers import set_seed

import warnings
warnings.filterwarnings('ignore')

# Local Library
import lib.SMART_globvars as gv
from models.build_ablation_study_model import get_demo_model, ablation_study_generation, calc_value_acc
from datasets_lib.build_dataset import get_dataset
from trainer.trainer_func import trainer_generate, code_generate

def eval():
    
    print('\n*****Schema and Argument Ablation Study.py Start*****')
    
    # model load...
    model, processor = get_demo_model(args.model_name, device)
    
    # test_data load...
    test_loader = get_dataset(args, processor)
    
    # Ablation Study Execute...
    result_dict = {}
    TP, ALL = 0, 0
    save_root = '/data/jhpark_checkpoint/schema_and_argument_ckpt/ablation_study'
    result_save_path = os.path.join(save_root, f'{args.model_name}_level{args.ablation_level}_30300.json')
    with open(result_save_path,'r') as r:
            result_dict = json.load(r)
            already_exist_puzzle = list(result_dict.keys())
    
    for i, (im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code) in tqdm(enumerate(test_loader)):
        
        if im_name in already_exist_puzzle:
            print(f'{i}idx, {im_name} skipped')
            continue
        
        question, pred = ablation_study_generation(args.model_name, args.ablation_level, model, processor,
                                           im_path, q_stn_out, Answer_Option_phrase, img_dcp, pseudo_code, device)
        
        TP, ALL, result_dict = calc_value_acc(pred, value_answer, im_name, question, TP, ALL, result_dict)
        
        if i % 300 == 0:
            with open(result_save_path,'w') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4) 
        
    print('\n*****Schema and Argument Ablation_study.py Complete*****')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Schema and Argument Code Prompting Eval.py")
    
    # Common arguments...
    parser.add_argument("--mode", type=str, default="ablation_study")
    parser.add_argument("--data", type=str, default="SMART") # [SMART, Geo3K]
    parser.add_argument("--model_name", type=str, default="GPT4o")    
    parser.add_argument("--data_root", type=str, default="/data/SMART101-release-v1/SMART101-Data/")
    
    # Train arguments...
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--loss_type", type=str, default="classifier")
    parser.add_argument("--load_ckpt_path", type=str, default="demo/epoch_2/")
    parser.add_argument("--use_gpu", type=str, default="0")
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1123)
    
    # Experiment arguments...
    parser.add_argument("--ablation_level", type=int, default=1) # [supervised, zeroshot]
    parser.add_argument("--experiment", type=str, default='supervised') # [supervised, zeroshot]
    parser.add_argument("--answer_type", type=str, default='value') # [option, value]
    parser.add_argument("--use_option_prompt", type=bool, default=False)
    parser.add_argument("--use_img", type=bool, default=False)
    
    # Path argumnets...
    parser.add_argument("--save_root", type=str, default='/data/jhpark_checkpoint/schema_and_argument_ckpt')
    parser.add_argument("--save_folder", type=str, default="None")
    parser.add_argument("--img_dcp_path", type=str, default="puzzle_img_dcp_101.json")
    parser.add_argument("--pseudo_code_path", type=str, default="puzzle_pseudo_code_101.json")
    
    args = parser.parse_args()
    gv.custom_globals_init()  
    set_seed(1123)
    
    global device
    device = f'cuda:{args.use_gpu}'
    print(device)
    eval()