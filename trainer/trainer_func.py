import os
import copy
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
import pytorch_warmup as warmup
from .logutil import init_logger, get_logger
from torch.optim.lr_scheduler import ExponentialLR
from lib.log_and_viz import *

def trainer_train(args, model, processor, train_loader, valid_loader):
    
    model.train()
    logger, save_folder = get_custom_logger(args)
    
    epochs = args.epochs   
    
    train_loss_logger, valid_loss_logger, lr_logger = dict(), dict(), dict()
    len_train, len_valid = len(train_loader), len(valid_loader)
    grad_accumulation_steps = 4
    valid_frequency_steps = 1000
    
    optimizer = AdamW([{'params': model.model.parameters(), 'lr': args.lr},
                       {'params': model.MCA1.parameters(), 'lr': args.lr*100},
                       {'params': model.MCA2.parameters(), 'lr': args.lr*100},
                       {'params': model.MCA3.parameters(), 'lr': args.lr*100}])
    
    # Scheduler and Warmup 
    if args.use_scheduler:
        lr_update_frequency_steps = int(len_train//10)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
        lr_logger[1] = scheduler.get_lr()[-1]
    
    # Train & Valid Integrated Loop...
    for epoch in tqdm(range(epochs)):
        
        steps = 0
        accumulated_avg_loss = 0
        
        # Train Loop bundle...
        for batch in train_loader:
            
            # if steps==5:break
            
            steps += 1
            inputs, labels = batch
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accumulation_steps
            
            accumulated_avg_loss += loss.item()
            loss.backward()
            
            logging_step = steps+(len_train*epoch)
            if steps % grad_accumulation_steps == 0:
                
                logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, training loss of previous {grad_accumulation_steps} batches: {accumulated_avg_loss:.8f}")
                train_loss_logger[logging_step] = accumulated_avg_loss
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
                
            # Valid Loop bundle...
            if steps % valid_frequency_steps == 0:
                
                model.eval()
                accumulated_valid_avg_loss = 0
                for v_batch in valid_loader:
                    v_inputs, v_labels = v_batch
                    
                    v_outputs = model(**v_inputs, labels=v_labels)
                    valid_loss = v_outputs.loss
                    
                    accumulated_valid_avg_loss += valid_loss.item()
                    
                accumulated_valid_avg_loss = accumulated_valid_avg_loss / len_valid
                valid_loss_logger[logging_step] = accumulated_valid_avg_loss
                logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, validation loss: {accumulated_valid_avg_loss:.8f}")
                model.train()
                
                # 학습 결과 JSON 결과 파일 저장
                train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
                with open(train_loss_save_path,'w') as f:
                    json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
                
                valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
                with open(valid_loss_save_path,'w') as f:
                    json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
                
                # 학습 결과 Plot 시각화 파일 저장
                loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
                plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
                
            # LR Scheduler Update...
            if args.use_scheduler:
                if steps % lr_update_frequency_steps == 0:
                    
                    past_lr = scheduler.get_lr()[-1]
                    with warmup_scheduler.dampening():
                        scheduler.step()
                    current_lr = scheduler.get_lr()[-1]
                    lr_logger[logging_step] = current_lr
                    logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, lr updated: {past_lr:.12f} --> {current_lr:.12f}")            
                    
        # 에폭별로 폴더를 만들기 위한 경로 설정
        epoch_output_dir = os.path.join(save_folder, f"epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        # 에폭별로 모델과 프로세서를 저장
        model.eval()
        save_model_path = os.path.join(epoch_output_dir, 'whole_model.pth')
        save_model = copy.deepcopy(model)
        torch.save(save_model.state_dict(), save_model_path)
        print(f'Save model: epoch {epoch+1} to {save_model_path}')
        del save_model
        model.train()
        
        model.save_pretrained(epoch_output_dir)
        processor.save_pretrained(epoch_output_dir)
        
        # 학습 결과 JSON 결과 파일 저장
        train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
        with open(train_loss_save_path,'w') as f:
            json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
        
        valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
        with open(valid_loss_save_path,'w') as f:
            json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
        
        # 학습 결과 Plot 시각화 파일 저장
        loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
        plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
        
        # 학습률 변화 결과 및 시각화 파일 저장
        if args.use_scheduler:
            lr_logger_save_path = os.path.join(args.save_root, args.save_folder, 'lr_logger.json')
            with open(lr_logger_save_path,'w') as f:
                json.dump(lr_logger, f, ensure_ascii=False, indent=4)       
                
            lr_curve_path = os.path.join(args.save_root, args.save_folder, 'lr_scheduler_curve.png')
            plot_lr_loss(lr_logger, epoch+1, len_train, lr_curve_path)
                
        write_chat_template(processor, epoch_output_dir, logger)
                
def get_custom_logger(args):
    save_folder = os.path.join(args.save_root, args.save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    init_logger(save_folder)
    logger = get_logger()
    return logger, save_folder

def write_chat_template(processor, output_dir, logger):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")
        
def trainer_generate(args, model, processor, test_loader):
    
    model.eval()
    TP, ALL = 0, 0
    result_dict = dict()
    for batch in test_loader:
        
        inputs, gt, im_name_list, question_list = batch
        
        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Evaluation
        if args.answer_type == 'option':
            TP, ALL, result_dict = calc_option_acc(output_texts, gt, im_name_list, TP, ALL, result_dict)
        elif args.answer_type == 'value':
            TP, ALL, result_dict = calc_value_acc(output_texts, gt, im_name_list, TP, ALL, result_dict)

    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.experiment}_{args.pseudo_code_type}_PC_Option_{args.use_option_prompt}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 
        
def trainer_viz_cross_attmap(args, model, processor, test_loader):
    
    model.eval()
    TP, ALL = 0, 0
    result_dict = dict()
    
    code_start = 151650
    code_end = 151651
    dcp_start = 151646
    dcp_end = 151647
    
    for batch in test_loader:
        
        inputs, gt, im_name_list, question_list = batch        
        
        # Attention Map Forward
        outputs = model(**inputs)
        input_ids = inputs['input_ids']
        attention_map3 = outputs['attentions']
        
        print(attention_map3.shape)
        B, H = attention_map3.shape[0], attention_map3.shape[1]        
        for i in range(B):
            for j in range(H):
                img_dcp_start_idx = (input_ids[i] == dcp_start).nonzero()[0]
                img_dcp_end_idx = (input_ids[i] == dcp_end).nonzero()[0]
                dcp_id = input_ids[i][img_dcp_start_idx+1:img_dcp_end_idx]
                dcp_len = (img_dcp_end_idx - img_dcp_start_idx -1)
                dcp_text = processor.batch_decode(
                    dcp_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                pseudo_start_idx = (input_ids[i] == code_start).nonzero()[0]
                pseudo_end_idx = (input_ids[i] == code_end).nonzero()[0]
                code_id = input_ids[i][pseudo_start_idx+1:pseudo_end_idx]
                code_len = (pseudo_end_idx - pseudo_start_idx -1)
                code_text = processor.batch_decode(
                    code_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                visualize_attention_map(args, attention_map3[i,j,:dcp_len,:code_len], dcp_text, code_text, head=j, puzzle_name = im_name_list[i])
        # break

def code_generate(args, model, processor, train_loader, valid_loader, test_loader):
    
    model.eval()
    result_dict = dict()
    for target_loader in [train_loader, valid_loader, test_loader]:
        for idx, batch in tqdm(enumerate(target_loader)):

            inputs, gt, puzzle_name, question_list = batch
            
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            for name, pred, q in zip(puzzle_name, output_texts, question_list):
                result_dict[name] = {'question': q,
                                    'pred': pred}
            
            print(f'batch {idx}/{len(target_loader)}: Pid: {name}, Question: {q}\nPred: {pred}\n')
        
    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.data}_{args.experiment}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 

def calc_option_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name in zip(pred_list, label_list, im_name_list):
        eval_answer = pred.upper()
        
        find_std_format1 = eval_answer[-100:].find(f'ANSWER: {label[-1]}')
        find_std_format2 = eval_answer[-100:].find(f'ANSWER:{label[-1]}')
        find_std_format3 = eval_answer[-100:].find(f'ANSWER.{label[-1]}')
        find_std_format4 = eval_answer[-100:].find(f'ANSWER. {label[-1]}')
        find_std_format5 = eval_answer[-100:].find(f'ANSWER IS {label[-1]}')      
          
        find_only_answer1 = (len(eval_answer)==1 and eval_answer[0] == label[-1])    
        find_only_answer2 = (eval_answer[:2] == f'{label[-1]}.')     
        find_only_answer3 = (eval_answer[-2:] == f' {label[-1]}') 
        find_only_answer4 = (eval_answer[-2:] == f':{label[-1]}')
        find_only_answer5 = (eval_answer[-2:] == f'.{label[-1]}')
        find_only_answer6 = (eval_answer[-2:] == f'\n{label[-1]}')
        find_only_answer7 = (eval_answer[-3:] == f' {label[-1]}.')
    
        if ((find_only_answer1) or (find_only_answer2) or (find_only_answer3) or (find_only_answer4) or (find_only_answer5) or (find_only_answer6) or (find_only_answer7) or\
            (find_std_format1>=0) or (find_std_format2>=0) or (find_std_format3>=0) or (find_std_format4>=0) or (find_std_format5>=0)):
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label}")
    except: pass
    return TP, ALL, result_dict

def calc_value_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name in zip(pred_list, label_list, im_name_list):
        eval_answer = pred.upper()
        eval_label = label.upper()
        
        find_answer1 = eval_answer.find(f' {eval_label}')
        find_answer2 = eval_answer.find(f'\n{eval_label}')
        find_answer3 = eval_answer.find(f':{eval_label}')
    
        if (find_answer1 >= 0) or (find_answer2 >= 0) or (find_answer3 >= 0):
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label},  img_name: {img_name}")
    except: pass
    return TP, ALL, result_dict