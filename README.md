# Code prompting based on schema and argument structure in Multimodal Visual Chain-of-Thought

## Code Execute Explain
- 사용 데이터는 현재 SMART 혹은 Geo3K이며, 모델은 Qwen2-VL-2B 사용중이다.
- 메인 실행 파일은 train.py, eval.py 이다.
- argument는 모두 위 두 파일에서 관리한다.
- GPU 세팅은 CUDA_VISIBLE_DEVICES 설정하고, use_gpu를 따라서 설정해주어여 한다. (Ex, CUDA_VISIBLE_DEVICES=1,2,3 --> --use_gpu = 0,1,2)

## SMART Script
```bash
# Train
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_wo_opt_dcp_mca --use_gpu 0,1,2 --use_img True

# Eval
CUDA_VISIBLE_DEVICES=3 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --load_ckpt_path value_gen_wo_opt_dcp_mca/epoch_1/whole_model.pth
```

## Geo3K Script
```bash
# Train
CUDA_VISIBLE_DEVICES=1,2 python train.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --save_folder geo3k_with_opt_dcp_mca --use_gpu 0,1 --use_img True --use_option_prompt True

# Eval
CUDA_VISIBLE_DEVICES=0 python eval.py --data Geo3K --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path geo3k_with_opt_dcp_mca/epoch_1/whole_model.pth
```