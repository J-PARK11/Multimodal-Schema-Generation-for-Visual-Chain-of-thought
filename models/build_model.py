import os
import torch
from peft import LoraConfig
from transformers import AutoProcessor
from models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLForConditionalClassfication
from models.schema_head import Schema_head_predictior

def get_model(args):
    
    if args.model_name == 'schema_head':
        model = Schema_head_predictior(args)
        processor = model.preprocess
        model.to('cuda')
        print('\n*****Build schema_head model*****')
        
        if args.mode == 'schema_head_test':
            model.load_state_dict(torch.load(args.load_ckpt_path, map_location='cuda'))            
            print(f'Load Schemad Head CKPT: {args.load_ckpt_path}')
        
    if args.model_name == 'Qwen2_VL_7B':
        
        use_gpu = args.use_gpu.split(',')
        max_memory={
            int(gpu_num): "40GiB" for gpu_num in use_gpu
        }
        
        # Train Setting
        pretrained_path = "Qwen/Qwen2-VL-7B-Instruct"
        if args.load_ckpt_path == 'None':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto", # [balanced, auto]
                max_memory=max_memory) 
            
            print('\n*****Build Pretrained Qwen2_VL Model*****')
            print(f'Load ckpt path: {pretrained_path}..')
        
        # Generation or Evaluation Setting
        else:
            load_ckpt_path = os.path.join(args.save_root, args.load_ckpt_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                load_ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto", # [balanced, auto]
                max_memory=max_memory)
            
            print('\n*****Build Trained Qwen2_VL Model*****')
            print(f'Load ckpt path: {args.load_ckpt_path}..')

        processor = AutoProcessor.from_pretrained(pretrained_path, min_pixels=128*28*28, max_pixels=256*28*28)
        print(f'Load Processor: {pretrained_path}..')
        print(f'Available GPU num: {use_gpu}....')
    
    elif args.model_name == 'Qwen2_VL_2B':
        
        use_gpu = args.use_gpu.split(',')
        max_memory={
            int(gpu_num): "40GiB" for gpu_num in use_gpu
        }
        
        # Train Setting
        pretrained_path = "Qwen/Qwen2-VL-2B-Instruct"
        warmup_state_path = "/data/jhpark_checkpoint/schema_and_argument_ckpt/geo3k_with_opt_dcp_mca/epoch_5/whole_model.pth"
        if args.load_ckpt_path == 'None':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="balanced", # [balanced, auto]
                max_memory=max_memory) 
            # model.load_state_dict(torch.load(warmup_state_path, map_location='cuda'))
            
            print('\n*****Build Pretrained Qwen2_VL Model*****')
            print(f'Load ckpt path: {pretrained_path}..')
            # print(f'Load Model State Warmup...: {warmup_state_path}..')
        
        # Generation or Evaluation Setting
        else:
            load_ckpt_path = os.path.join(args.save_root, args.load_ckpt_path)
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="balanced", # [balanced, auto]
                max_memory=max_memory)
            model.load_state_dict(torch.load(load_ckpt_path, map_location='cuda'))
            
            print('\n*****Build Trained Qwen2_VL Model*****')
            print(f'Load ckpt path: {args.load_ckpt_path}..')

        processor = AutoProcessor.from_pretrained(pretrained_path, min_pixels=128*28*28, max_pixels=256*28*28)
        print(f'Load Processor: {pretrained_path}..')
        print(f'Available GPU num: {use_gpu}....')
        
    for name, param in model.named_parameters():
        if 'visual' in name:
            param.requires_grad=False
    
    print(f'\n{model}')
    print(f'\nRequire Grad Parameter numbers: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Freeze Parameter numbers: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}')
    
    return model, processor
