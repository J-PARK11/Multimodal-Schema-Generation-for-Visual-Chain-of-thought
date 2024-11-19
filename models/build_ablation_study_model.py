import os
import copy
import json
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from models.meteor.arch.meteor_utils import freeze_model

# from huggingface_hub import login
# login(token = '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')
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from lib.llava_onevision_utils import conv_templates, SeparatorStyle

def get_demo_model(VLM_type, device):
    processor = None
    
    if VLM_type == 'GPT4o':
        from openai import OpenAI, BadRequestError
        api_key = 'Fill API Key'
        model = OpenAI(api_key=api_key)
        print(f'\nGPT4o Model Used') 
    
    elif VLM_type == 'Meteor':
        from models.meteor.load_mmamba import load_mmamba
        from models.meteor.load_meteor import load_meteor
        mmamba_path = 'BK-Lee/Meteor-Mamba'
        mlm_path = 'BK-Lee/Meteor-MLM'
        mmamba = load_mmamba(mmamba_path).to(device)
        model, processor = load_meteor(mlm_path, bits=None)
        model.to(device)
        
        freeze_model(mmamba)
        freeze_model(model)

    elif VLM_type == 'Llava_onevision':
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from models.llava_onevision.llava_onevision_utils import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        
        model_name = "llava_qwen"
        pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        device_map = "auto"
        
        processor, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)   
    
    if VLM_type == 'Meteor':
        model = [mmamba, model]
    elif VLM_type == 'Llava_onevision':
        processor = [processor, image_processor, max_length]

    return model, processor

def ablation_study_generation(VLM_type, ablation_level, model, processor, im_path, q_stn_out, Answer_Option_phrase, img_dcp, pseudo_code, device):
    
    # im_path = im_path[0]
    # question = q_stn_out[0]
    # img_dcp = img_dcp[0]
    # pseudo_code = pseudo_code[0]
    
    if ablation_level == 1:
        api_key = 'Fill API Key'
        instruction_prompt = "Please solve the problem using the question and image. And return in only answer value."
        question = f'Question: {q_stn_out}\nOptions: {Answer_Option_phrase}\nInstruction: {instruction_prompt}'

    elif ablation_level == 2:
        api_key = 'Fill API Key'
        instruction_prompt = "Please solve the problem using the question, image and pseudo code provided. And return in only answer value."
        question = f'Pseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {Answer_Option_phrase}\nInstruction: {instruction_prompt}'
        
    elif ablation_level == 3:
        api_key = 'Fill API Key'
        instruction_prompt = "Please solve the problem using the question, image, image description, and pseudo code provided. And return in only answer value."
        question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {Answer_Option_phrase}\nInstruction: {instruction_prompt}'
    
    if VLM_type == 'GPT4o':
        
        import base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
        try:
            gpt_img_format = encode_image(im_path)
            response = model.chat.completions.create(            
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}
                        ],
                    },                
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gpt_img_format}"}},
                        {"type": "text", "text": question},
                    ]}
                ],
            temperature=0.1, max_tokens=800)
            decoded_text = response.choices[0].message.content.strip().replace('\n\n', '\n').replace('\\', '').replace('   ', ' ')
        except:
            decoded_text = 'None'
            print('GPT ERROR!!')
        
    elif VLM_type == 'Meteor':
                
        # Model & I/O Setting
        mmamba, meteor = model[0], model[1]
        image_token_number = int((490/14)**2)
        image = F.interpolate(pil_to_tensor(Image.open(im_path).convert("RGB")).unsqueeze(0), size=(490, 490)).squeeze(0)
        inputs = [{'image': image, 'question': question}]
        
        # Meteor Mamba
        mmamba_inputs = mmamba.eval_process(inputs=inputs, tokenizer=processor, device=device, img_token_number=image_token_number)
        if 'image' in mmamba_inputs.keys():
            clip_features = meteor.clip_features(mmamba_inputs['image'])
            mmamba_inputs.update({"image_features": clip_features})
        
        with torch.no_grad():
            mmamba_outputs = mmamba(**mmamba_inputs)
        
        # Meteor
        meteor_inputs = meteor.eval_process(inputs=inputs, data='demo', tokenizer=processor, device=device, img_token_number=image_token_number)
        if 'image' in mmamba_inputs.keys():
            meteor_inputs.update({"image_features": clip_features})
        meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})    

        # Generation
        with torch.no_grad():
            generate_ids = meteor.generate(**meteor_inputs, do_sample=True, max_new_tokens=500, top_p=0.95, temperature=0.9, use_cache=True)
            decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('assistant\n')[-1].split('[U')[0].strip()
            
            
    elif VLM_type == 'Llava_onevision':
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from models.llava_onevision.llava_onevision_utils import conv_templates, SeparatorStyle
        
        # Model & I/O Setting
        processor, image_processor, max_length = processor[0], processor[1], processor[2]
        image = Image.open(im_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        
        conv_template = "qwen_1_5"  
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, processor, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]
        
        # Generation
        with torch.no_grad():
            generate_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=4096)
            decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            
    return question, decoded_text

def calc_value_acc(pred, label, im_name, question, TP, ALL, result_dict):
    
    eval_answer = pred.upper()
    eval_label = label.upper()
    
    find_answer1 = (eval_answer == eval_label)
    
    find_answer2 = eval_answer.find(f' {eval_label}')
    find_answer3 = eval_answer.find(f'\n{eval_label}')
    find_answer4 = eval_answer.find(f':{eval_label}')
    find_answer5 = eval_answer.find(f'*{eval_label}')
    find_answer6 = eval_answer.find(f'({eval_label}')

    if (find_answer1) or (find_answer2 >= 0) or (find_answer3 >= 0) or (find_answer4 >= 0) or (find_answer5 >= 0) or (find_answer6 >= 0):
        TP += 1   
        hit = True
    else:
        hit = False    
    ALL += 1
    
    result_dict[im_name] = {
        'Pred': pred,
        'Label': label,
        'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: 
        print(f"\nAccuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label},  img_name: {im_name}")
    except: pass
    return TP, ALL, result_dict
