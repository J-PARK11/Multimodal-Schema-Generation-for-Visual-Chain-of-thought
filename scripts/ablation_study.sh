# GPT4o
python ablation_study.py --model_name GPT4o --ablation_level 1 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
python ablation_study.py --model_name GPT4o --ablation_level 2 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
python ablation_study.py --model_name GPT4o --ablation_level 3 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0

# METEOR
python ablation_study.py --model_name Meteor --ablation_level 1 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
python ablation_study.py --model_name Meteor --ablation_level 2 --experiment supervised --answer_type value --batch_size 1 --use_gpu 1
python ablation_study.py --model_name Meteor --ablation_level 3 --experiment supervised --answer_type value --batch_size 1 --use_gpu 2

# Llava_onevision
CUDA_VISIBLE_DEVICES=3 python ablation_study.py --model_name Llava_onevision --ablation_level 1 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
CUDA_VISIBLE_DEVICES=0 python ablation_study.py --model_name Llava_onevision --ablation_level 2 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
CUDA_VISIBLE_DEVICES=1 python ablation_study.py --model_name Llava_onevision --ablation_level 3 --experiment supervised --answer_type value --batch_size 1 --use_gpu 0
