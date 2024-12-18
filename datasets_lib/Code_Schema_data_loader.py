import os
import pdb
import math
import json
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from PIL import Image
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset
from transformers.image_utils import load_image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from models.qwen2_vl.qwen_vl_utils import process_vision_info

import lib.utils as utils
import lib.SMART_globvars as gv

class V_COT_SMART101_Dataset(Dataset):
    def __init__(self, args, mode):
        
        self.args = args
        self.mode = mode
        self.experiment = args.experiment
        self.diff = 'easy'
        self.data_root = args.data_root
        
        # Image Description & Pseudo Code Loading...
        self.image_description_path = os.path.join('/data/SMART101-release-v1/schema_and_argument_data',
                                                   args.img_dcp_path)
        self.pseudo_code_path = os.path.join('/data/SMART101-release-v1/schema_and_argument_data',
                                             args.pseudo_code_path)
        self.schema_head_prediction_path = os.path.join('/data/jhpark_checkpoint/schema_and_argument_ckpt/',
                                                        'schema_head_lr_1e3/epoch_1/schema_classifier_head_smart_testset_result.json')
    
        with open(self.image_description_path,'r') as f:
            self.image_description = json.load(f)
            print(f'\nImage Description: {len(self.image_description)}')

        with open(self.pseudo_code_path,'r') as f:
            self.pseudo_code = json.load(f)    
            print(f'Pseudo Code: {len(self.pseudo_code)}\n')          
            
        with open(self.schema_head_prediction_path,'r') as f:
            self.schema_head_pred = json.load(f)    
            print(f'Pseudo schema_head_pred: {len(self.schema_head_pred)}\n')       
            
        # Supervised vs Zeroshot Experiment Setting Root Puzzle Split...
        if self.mode == 'train':
            if self.experiment == 'supervised':
                self.puzzle_ids = [str(pid) for pid in range(1,102)]
            elif self.experiment == 'zeroshot':
                self.puzzle_ids = [pid for pid in range(1,102)]
                self.puzzle_ids = sorted(list(set(self.puzzle_ids) - set(gv.PS_VAL_IDX) - set(gv.PS_TEST_IDX)))
                self.puzzle_ids = list(map(str, self.puzzle_ids))
            self.start_idx = 0
            self.end_idx = 1600
                                
        elif self.mode == 'valid':
            if self.experiment == 'supervised':
                self.puzzle_ids = [str(pid) for pid in range(1,102)]
            elif self.experiment == 'zeroshot':
                self.puzzle_ids = gv.PS_VAL_IDX
                self.puzzle_ids = list(map(str, self.puzzle_ids))
            self.start_idx = 1600
            self.end_idx = 1620 # 1700
            
        elif self.mode in ['test', 'ablation_study']:
            if self.experiment == 'supervised':
                self.puzzle_ids = [str(pid) for pid in range(1,102)]
            elif self.experiment == 'zeroshot':
                self.puzzle_ids = gv.PS_TEST_IDX
                self.puzzle_ids = list(map(str, self.puzzle_ids))
            
            # 강현씨 정성평가
            # self.puzzle_ids = ['17']
            self.start_idx = 1700
            self.end_idx = 2000
        
        elif self.mode == 'viz_attmap':
            if self.experiment == 'supervised':
                self.puzzle_ids = [str(pid) for pid in range(1,102)]
            elif self.experiment == 'zeroshot':
                self.puzzle_ids = gv.PS_TEST_IDX
                self.puzzle_ids = list(map(str, self.puzzle_ids))            
            self.start_idx = 1700
            self.end_idx = 1701
        
        # Get Puzzle Instance by Root Puzzle Split...
        self.qa_info = self.Get_Puzzle_Instance()
        self.num_tot = self.end_idx-self.start_idx
        print(f'{self.mode}, {self.experiment} Num of Root: {len(self.puzzle_ids)},  Num of Instance per root: {self.num_tot} = {len(self.puzzle_ids)*self.num_tot}')
        
    def Get_Puzzle_Instance(self):
        qa_bundle = []
            
        # 인스턴스 퍼즐 불러오기.
        for puzzle_id in self.puzzle_ids:
            
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            
            qa_info = qa_info[self.start_idx:self.end_idx]
                
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
                
            qa_bundle = qa_bundle + qa_info
            
        return qa_bundle
        
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im_name = im_path.split('/')[-1]        
        
        # im = load_image(im_path) #.resize((912,912))
        q_stn = info["Question"]
                
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(gv.MAX_DECODE_STEPS)
        
        # 시퀀스 데이터 예외처리.
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                pdb.set_trace()
        
        if not info: info = []
        opts = []
        Answer_Option_phrase = ''
        for op in ["A", "B", "C", "D", "E"]:
            op_val = info[op]
            Answer_Option_phrase += f' {op_val},'
            # Answer_Option_phrase += f' {op}. {op_val}'
            opts.append(op_val)
        
        q_stn_out = q_stn
        option_answer= f'Answer: {info["Answer"]}' # info['Answer']
        value_answer = opts[lbl]
        Answer_Option_phrase = Answer_Option_phrase.strip()
                
        # Get Image Description & Pseudo Code
        img_dcp = self.image_description[im_name]
        if self.args.pseudo_code_type == 'GT':
            pseudo_code = self.pseudo_code[f'puzzle_{pid}']
        elif self.args.pseudo_code_type == 'schema_head':
            pred_pid = self.schema_head_pred[im_name]['Pred']
            print(pid, pred_pid)
            pseudo_code = self.pseudo_code[f'puzzle_{pred_pid}']
                
        return im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code

    def __len__(self):
        return len(self.qa_info)

def img_train_collator_fn(data, args, processor, device):
    
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        
        if args.answer_type == 'option':
            instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
            label = option_answer
        elif args.answer_type == 'value':
            instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided."
            label = value_answer
        
        if args.use_option_prompt:
            question = f'Description of image: <|object_ref_start|>{img_dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{pseudo_code}<|quad_end|>\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: <|object_ref_start|>{img_dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{pseudo_code}<|quad_end|>\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
        
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im_path},
                    {'type': 'text', 'text': question}]}, 
            
            {   'role': 'assistant', 'content': [{'type': 'text', 'text': f'{label}'}]}]
        
        messages.append(prompt)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def img_test_collator_fn(data, args, processor, device):
    
    if args.answer_type == 'option':
        instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
    elif args.answer_type == 'value':
        instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided."
    
    gt = []
    im_name_list = []
    messages = []
    question_list = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:

        if args.use_option_prompt:
            question = f'Description of image: <|object_ref_start|>{img_dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{pseudo_code}<|quad_end|>\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: <|object_ref_start|>{img_dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{pseudo_code}<|quad_end|>\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
            
        # 강현씨 정성평가
        # instruction_prompt = "Please provide the reasoning steps along with the answer. The solution process must be included."
        # question = f'Description: <|object_ref_start|>{img_dcp}<|object_ref_end|>\n pseudocode: <|quad_start|>{pseudo_code}<|quad_end|>\nOption: {options} \nQuestion: {q_stn_out}\n  Instruction: {instruction_prompt} "Please explain each step in detail before providing the answer"\n\n Example: \n 1. **Step 1**:\n...\n3. **Conclusion**:'
        
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im_path},
                    {'type': 'text', 'text': question}]}
            ]
        
        messages.append(prompt)
        gt.append(value_answer)
        im_name_list.append(im_name)
        question_list.append(question)
    
    # print(question)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    return inputs, gt, im_name_list, question_list
    

def img_dcp_train_collator_fn(data, args, processor, device):
    
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        
        if args.answer_type == 'option':
            instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
            label = option_answer
        elif args.answer_type == 'value':
            instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
            label = value_answer
        
        if args.use_option_prompt:
            question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
        
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}, 
            
            {   'role': 'assistant', 'content': [{'type': 'text', 'text': f'{label}'}]}]
        
        messages.append(prompt)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def img_dcp_test_collator_fn(data, args, processor, device):
    
    if args.answer_type == 'option':
        instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
    elif args.answer_type == 'value':
        instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
    
    gt = []
    im_name_list = []
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:

        if args.use_option_prompt:
            question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
            
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}
            ]
        
        messages.append(prompt)
        gt.append(value_answer)
        im_name_list.append(im_name)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    return inputs, gt, im_name_list

def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))