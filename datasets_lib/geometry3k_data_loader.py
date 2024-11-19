import os
import sys
import json
import torch
import codecs
from PIL import Image
from pathlib import Path

from models.qwen2_vl.qwen_vl_utils import process_vision_info

class Geometry3KDataset(torch.utils.data.Dataset):
    
    def __init__(self, args, mode):
        
        self.mode = mode
        self.data_root = '/data/METEOR/geo3k'
        self.opt_to_value = {'A':0, 'B': 1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
        self.opt_list = list(self.opt_to_value.keys())
        
        self.geo3k_pseudo_code_path = os.path.join("/data/SMART101-release-v1/schema_and_argument_data","kkh_made_geo3k_pseudo_code.json")
        with open(self.geo3k_pseudo_code_path,'r') as f:
            self.geo3k_pseudo_code = json.load(f)    
            print(f'geo3k_pseudo_code: {len(self.geo3k_pseudo_code)}\n')            
        
        # 학습, 검증, 시험 Path Definition
        if self.mode == 'train':
            self.data_folder_path = os.path.join(self.data_root, 'train')  
            self.puzzle_ids = [str(pid) for pid in range(0,2101)]      
        elif self.mode == 'valid':
            self.data_folder_path = os.path.join(self.data_root, 'images', 'geo3k', 'val')
            self.puzzle_ids = [str(pid) for pid in range(2101,2401)]
        elif self.mode == 'test':
            self.data_folder_path = os.path.join(self.data_root, 'images', 'geo3k', 'test')
            self.puzzle_ids = [str(pid) for pid in range(2401,3002)]
        self.data_len = len(self.puzzle_ids)
        
        print(self.data_folder_path)
        print(f'Geo3K Dataset {self.mode} mode, datalen: {self.data_len}')    
            
    def __getitem__(self, index):
        ins_id = self.puzzle_ids[index]
        ins_path = os.path.join(self.data_folder_path, ins_id)
        
        im_path = os.path.join(ins_path, 'img_diagram.png')
        annot_path = os.path.join(ins_path, 'data.json')        
        logic_path = os.path.join(ins_path, 'logic_form.json')        
        
        with open(annot_path,'r') as f:
            annot_data = json.load(f)
        with open(logic_path,'r') as h:
            logic_form_data = json.load(h)
            
        q = annot_data['problem_text']
        opts = annot_data['choices']
        option_answer = f"Answer: {annot_data['answer']}"
        value_answer = f"Answer: {opts[self.opt_to_value[annot_data['answer']]]}"
        opts_prompt = ''
        for ii in range(len(opts)):
            op_val = opts[ii]
            opts_prompt += f' {self.opt_list[ii]}. {op_val}'
        
        # Schema
        # schema = logic_form_data['text_logic_form']
        try:
            schema = self.geo3k_pseudo_code[ins_id]['pseudo_code'][:800]
        except:
            schema = "None of Pseudo Code"
                
        dcp_logic = logic_form_data['diagram_logic_form']
        dcp_pos = logic_form_data['point_positions']
        dcp_line = logic_form_data['line_instances']
        dcp = f'{dcp_logic}\nPoint positions: {dcp_pos}\nline instances: {dcp_line}'
              
        return ins_id, im_path, q, opts, opts_prompt, option_answer, value_answer, schema, dcp #, schema, dcp
    
    def __len__(self):
        return self.data_len
    
def img_train_collator_fn(data, args, processor, device):
    
    messages = []
    for ins_id, im_path, q, opts, opts_prompt, option_answer, value_answer, schema, dcp in data:
        
        if args.answer_type == 'option':
            instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
            label = option_answer
        elif args.answer_type == 'value':
            instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided."
            label = value_answer
        
        if args.use_option_prompt:
            question = f'Description of image: <|object_ref_start|>{dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{schema}<|quad_end|>\nQuestion: {q}\nOptions:{opts_prompt}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: <|object_ref_start|>{dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{schema}<|quad_end|>\nQuestion: {q}\nInstruction: {instruction_prompt}'
        
        # 베이스라인: 질문과 이미지만 넣기
        # question = f'Question: {q}\nOptions:{opts_prompt}\nInstruction: Please solve the problem.'
        
        # 크로스 어텐션 없이 수도코드와 디스크립터 넣고    
        # question = f'Description of image: {dcp}\nPseudo_code: {schema}\nQuestion: {q}\nOptions:{opts_prompt}\nInstruction: Please solve the problem using the question, image, image description, and pseudo code provided.'        
        
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
    # print(question)
    # print(label)
    # print()
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
    for ins_id, im_path, q, opts, opts_prompt, option_answer, value_answer, schema, dcp in data:

        if args.use_option_prompt:
            question = f'Description of image: <|object_ref_start|>{dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{schema}<|quad_end|>\nQuestion: {q}\nOptions:{opts_prompt}\nInstruction: {instruction_prompt}'
        else:
            question = f'Description of image: <|object_ref_start|>{dcp}<|object_ref_end|>\nPseudo_code: <|quad_start|>{schema}<|quad_end|>\nQuestion: {q}\nInstruction: {instruction_prompt}'
        
        # 베이스라인: 질문과 이미지만 넣기
        # question = f'Question: {q}\nOptions:{opts_prompt}\nInstruction: Please solve the problem.'
        
        # 크로스 어텐션 없이 수도코드와 디스크립터 넣고    
        # question = f'Description of image: {dcp}\nPseudo_code: {schema}\nQuestion: {q}\nOptions:{opts_prompt}\nInstruction: Please solve the problem using the question, image, image description, and pseudo code provided.'
        
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
        im_name_list.append(im_path.split('/')[-2])
        question_list.append(question)
        
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

def img_test_code_gen_collator_fn(data, args, processor, device):
    
    gt = []
    im_name_list = []
    messages = []
    question_list = []
    for ins_id, im_path, q, opts, opts_prompt, option_answer, value_answer, schema, dcp in data:
            
        instruction_prompt = "Please generate python-style step-by-step code to solve the question."        
        question = f'Logics of image: {dcp}\nQuestion: {q}\nPrompt: {instruction_prompt}'
        
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to make some python program to solve the question.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im_path},
                    {'type': 'text', 'text': question}]}
            ]
        
        messages.append(prompt)
        gt.append(value_answer)
        im_name_list.append(im_path.split('/')[-2])
        question_list.append(question)
        
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