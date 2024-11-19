import os
import copy
import json
import torch
import torch.nn as nn
from PIL import Image
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
    valid_frequency_steps = 500
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([{'params': model.qv_fusion.parameters(), 'lr': args.lr},
                       {'params': model.qvo_fusion.parameters(), 'lr': args.lr},
                       {'params': model.q_MLP.parameters(), 'lr': args.lr},
                       {'params': model.i_MLP.parameters(), 'lr': args.lr},
                       {'params': model.im_cnn.parameters(), 'lr': args.lr}])
    
    # Scheduler and Warmup 
    if args.use_scheduler:
        lr_update_frequency_steps = int(len_train//10)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
        lr_logger[1] = scheduler.get_lr()[-1]
    
    # Train & Valid Integrated Loop...
    for epoch in tqdm(range(epochs)):
        
        steps = 0
        
        # Train Loop bundle...
        for im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code in train_loader:
            
            # if steps==5:break
            steps += 1
            
            B = len(im_name)
            im_list, question_list, label_list = [], [], []
            for im_p, q, aop, p in zip(im_path, q_stn_out, Answer_Option_phrase, pid):
                prompt = f'Question: {q}\nOptions: {aop}'
                im = processor(Image.open(im_p).convert("RGB"))
                im_list.append(im)
                question_list.append(prompt)
                label_list.append(int(p)-1)
            im_list = torch.stack(im_list,0).to('cuda')
            label_list = torch.tensor(label_list).to('cuda')
            
            outputs = model(im_list, question_list)
            loss = criterion(outputs, label_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logging_step = steps+(len_train*epoch)
            logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, training loss: {loss:.8f}")
            train_loss_logger[logging_step] = float(loss)
            
            print(f'Pred: {torch.argmax(outputs, 1)}')
            print(f'Pred: {label_list}')
                
            # Valid Loop bundle...
            if steps % valid_frequency_steps == 1:
                
                # model.eval()
                # accumulated_valid_avg_loss = 0
                # v_steps=0
                # for im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code in valid_loader:
                    
                #     v_steps+=1               
                #     B = len(im_name)
                #     im_list, question_list, label_list = [], [], []
                #     for im_p, q, aop, p in zip(im_path, q_stn_out, Answer_Option_phrase, pid):
                #         prompt = f'Question: {q}\nOptions: {aop}'
                #         im = processor(Image.open(im_p).convert("RGB"))
                #         im_list.append(im)
                #         question_list.append(prompt)
                #         label_list.append(int(p))
                #     im_list = torch.stack(im_list,0).to('cuda')
                #     label_list = torch.tensor(label_list).to('cuda')
                    
                #     v_outputs = model(im_list, question_list)
                #     valid_loss = criterion(v_outputs, label_list)
                    
                #     print(v_steps, torch.argmax(v_outputs,1))
                #     print(label_list)
                    
                #     accumulated_valid_avg_loss += valid_loss
                    
                # accumulated_valid_avg_loss = accumulated_valid_avg_loss / len_valid
                # valid_loss_logger[logging_step] = float(accumulated_valid_avg_loss)
                # logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, validation loss: {float(accumulated_valid_avg_loss):.8f}")
                # model.train()
                
                # 학습 결과 JSON 결과 파일 저장
                train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
                with open(train_loss_save_path,'w') as f:
                    json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
                
                # valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
                # with open(valid_loss_save_path,'w') as f:
                #     json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
                
                # # 학습 결과 Plot 시각화 파일 저장
                # loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
                # plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
                
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
                
        # 학습 결과 JSON 결과 파일 저장
        train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
        with open(train_loss_save_path,'w') as f:
            json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
        
        # valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
        # with open(valid_loss_save_path,'w') as f:
        #     json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
        
        # # 학습 결과 Plot 시각화 파일 저장
        # loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
        # plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
        
        # 학습률 변화 결과 및 시각화 파일 저장
        if args.use_scheduler:
            lr_logger_save_path = os.path.join(args.save_root, args.save_folder, 'lr_logger.json')
            with open(lr_logger_save_path,'w') as f:
                json.dump(lr_logger, f, ensure_ascii=False, indent=4)       
                
            # lr_curve_path = os.path.join(args.save_root, args.save_folder, 'lr_scheduler_curve.png')
            # plot_lr_loss(lr_logger, epoch+1, len_train, lr_curve_path)
                                
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
    for im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code in test_loader:        
                
        B = len(im_name)
        im_list, question_list, label_list = [], [], []
        for im_p, q, aop, p in zip(im_path, q_stn_out, Answer_Option_phrase, pid):
            prompt = f'Question: {q}\nOptions: {aop}'
            im = processor(Image.open(im_p).convert("RGB"))
            im_list.append(im)
            question_list.append(prompt)
            label_list.append(int(p)-1)
        im_list = torch.stack(im_list,0).to('cuda')
        label_list = torch.tensor(label_list).to('cuda')
        
        outputs = model(im_list, question_list)        
        output_texts = torch.argmax(outputs, 1)
        
        # Evaluation
        TP, ALL, result_dict = calc_schema_head_acc(output_texts, label_list, im_name, TP, ALL, result_dict)

    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.experiment}_Option_{args.use_option_prompt}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 
        
def trainer_geo3k_generate(args, model, processor, train_loader, valid_loader, test_loader):
    
    model.eval()
    TP, ALL = 0, 0
    result_dict = dict()
    for target_loader in [train_loader, valid_loader, test_loader]:
        for ins_id, im_path, q, opts, opts_prompt, option_answer, value_answer in target_loader:        
                    
            B = len(im_path)
            im_list, question_list = [], []
            for im_p, q, aop in zip(im_path, q, opts_prompt):
                prompt = f'Question: {q}\nOptions: {aop}'
                im = processor(Image.open(im_p).convert("RGB"))
                im_list.append(im)
                question_list.append(prompt)
            im_list = torch.stack(im_list,0).to('cuda')
            
            outputs = model(im_list, question_list)        
            output_texts = torch.argmax(outputs, 1)
            
            for save_loop in range(len(ins_id)):  
                key = int(ins_id[save_loop])
                value = int(output_texts[int(save_loop)]+1)
                result_dict[key] = {'Schema_type': value}
                print(key, value)

        result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.experiment}_Geo3K_Option_{args.use_option_prompt}_result_dict.json')
        with open(result_save_path,'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4) 

def calc_schema_head_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name in zip(pred_list, label_list, im_name_list):
        
        pred = int(pred)+1
        label = int(label)+1
        
        if pred == label:
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {pred},  Label: {label},  img_name: {img_name}")
    except: pass
    return TP, ALL, result_dict