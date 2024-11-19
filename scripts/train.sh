# debugging
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --mode train --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --save_folder dump

# Value Generation with Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_supervised --use_gpu 0,1
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_zeroshot --use_gpu 2,3

# Value Generation w/o Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_generation_wo_option_prompt_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --save_folder value_generation_wo_option_prompt_zeroshot

# Option Generation
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type option --use_option_prompt True --save_folder option_generation_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type option --use_option_prompt True --save_folder option_generation_zeroshot

# On Air: Value Generation with Option Prompt Use Image and Image Dcp: GPU 0,1
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_wo_opt_dcp_mca --use_gpu 0,1,2 --use_img True
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_with_opt_dcp_mca --use_gpu 0,1,2 --use_img True --use_option_prompt True


CUDA_VISIBLE_DEVICES=3 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_1/whole_model.pth
CUDA_VISIBLE_DEVICES=3 python viz_attmap.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_wo_opt_dcp_mca/epoch_1/whole_model.pth


CUDA_VISIBLE_DEVICES=1,2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_wo_opt_dcp_mca_continue --use_gpu 0,1,2 --use_img True
CUDA_VISIBLE_DEVICES=2 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_3/whole_model.pth
CUDA_VISIBLE_DEVICES=3 python viz_attmap.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_3/whole_model.pth

CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --load_ckpt_path value_gen_wo_opt_dcp_mca/epoch_3/whole_model.pth

# SMART Real CA1,2
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 3 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_with_opt_dcp_real_mca --use_gpu 0,1 --use_img True --use_option_prompt True
CUDA_VISIBLE_DEVICES=0 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_with_opt_dcp_real_mca/epoch_2/whole_model.pth --pseudo_code_type schema_head

# Geo3K
CUDA_VISIBLE_DEVICES=0,1 python train.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --save_folder geo3k_baseline --use_gpu 0,1 --use_img True --use_option_prompt True
CUDA_VISIBLE_DEVICES=1 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_baseline/epoch_3/whole_model.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_baseline/epoch_5/whole_model.pth

CUDA_VISIBLE_DEVICES=0,1 python train.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --save_folder geo3k_kkh_pseudo_full_logic_dcp --use_gpu 0,1 --use_img True --use_option_prompt True
CUDA_VISIBLE_DEVICES=1 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_kkh_pseudo_full_logic_dcp/epoch_3/whole_model.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_kkh_pseudo_full_logic_dcp/epoch_5/whole_model.pth

CUDA_VISIBLE_DEVICES=0,1 python train.py --data Geo3K --model_name Qwen2_VL_2B --epochs 10 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --save_folder geo3k_kkh_pseudo_full_logic_dcp_x_attention --use_gpu 0,1 --use_img True --use_option_prompt True
CUDA_VISIBLE_DEVICES=0 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 10 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_kkh_pseudo_full_logic_dcp_x_attention/epoch_3/whole_model.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 10 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_kkh_pseudo_full_logic_dcp_x_attention/epoch_5/whole_model.pth
