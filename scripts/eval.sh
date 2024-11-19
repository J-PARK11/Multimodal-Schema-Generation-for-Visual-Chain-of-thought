python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path None
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_option_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_no_option_prompt_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B_Clf --load_ckpt_path qwen2_vl_2b_clf_batch_8_lr_1e5_epoch5/epoch_5

python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path None
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_option_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_no_option_prompt_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B_Clf --load_ckpt_path qwen2_vl_2b_clf_batch_8_lr_1e5_epoch5/epoch_5

CUDA_VISIBLE_DEVICES=0,1 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --pseudo_code_type schema_head --load_ckpt_path value_gen_wo_opt_dcp_mca/epoch_3/whole_model.pth
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --pseudo_code_type schema_head --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_1/whole_model.pth
CUDA_VISIBLE_DEVICES=2 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --pseudo_code_type schema_head --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_2/whole_model.pth
CUDA_VISIBLE_DEVICES=3 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 1 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --pseudo_code_type schema_head --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_4/whole_model.pth



CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Qwen2_VL_7B --data Geo3K --epochs 5 --batch_size 4 --lr 1e-5 --experiment code_gen_ft --answer_type value --use_gpu 0 --use_img True 