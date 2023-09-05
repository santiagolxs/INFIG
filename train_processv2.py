"""
Name: train_process
Date: 2022/4/11 上午10:26
Version: 1.0

"""

import torch
# from transformers import AdamW
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
import dev_processv2 as dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
# import tensorflow as tf
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import math


def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion, log_summary_writer:SummaryWriter=None, tokenizer=None, image_id_list=None):
    optimizer = None

    # pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
    #         "lr": 0,
    #     },
    #     {
    #         "params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param],
    #         "lr": opt.fuse_lr,
    #     }
    # ]


    # if opt.optim == 'adam':
    #     optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    # elif opt.optim == 'adamw':
    #     optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    # elif opt.optim == 'sgd':
    #     optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)
    if opt.optim =='adam':
        optimizer = Adam(cl_model.parameters(), lr=opt.fuse_lr,betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(cl_model.parameters(), lr=opt.fuse_lr,betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(cl_model.parameters(), momentum=opt.momentum)
    warm_up_iter = 10
    T_max = opt.epoch	# 周期
    lr_max = opt.fuse_lr	# 最大值
    lr_min = opt.lr	# 最小值
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/lr_max           #这里的0.1是初始的意思
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


    orgin_param = ModelParam()
    augment_param = ModelParam()
    # total_steps = len(train_loader) * opt.epoch/opt.acc_grad
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.15,num_training_steps=total_steps)
    # #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.15,num_training_steps=total_steps)

    scaler = GradScaler()
    last_F1 = 0
    last_Accuracy = 0
    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0

        cl_model.train()
        cl_model.zero_grad()

        # if epoch >= opt.train_fuse_model_epoch:
        #     optimizer.param_groups[0]['lr'] = opt.lr
        #     optimizer.param_groups[1]['lr'] = opt.lr


        # if epoch >= 60:                                     
        #     optimizer.param_groups[0]['lr'] = 8e-6
        #     optimizer.param_groups[1]['lr'] = 8e-6

        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:',position=0)
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(train_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels,\
                texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels,clip_to_id,clip_att_mask,clip_image = data

            local_rank = torch.distributed.get_rank()
            mcontext = cl_model.no_sync if local_rank != -1 and (index + 1) % opt.acc_grad  != 0 else nullcontext

            if opt.cuda is True:
                clip_to_id=clip_to_id.cuda()
                clip_image=clip_image.cuda()
                clip_att_mask=clip_att_mask.cuda()
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)

            with mcontext():
                with autocast(): 
                    origin_res, l_pos_neg, cl_lables, cl_self_loss,iamge_textsum,text_imagesum,loss_ita_hidden = cl_model(data_orgin=orgin_param, data_augment=augment_param, labels=labels, target_labels=target_labels,trainortest=1,clip_to_id=clip_to_id,clip_att_mask=clip_att_mask,clip_image=clip_image)
                    ground_truthtext_image = torch.arange(texts_origin.shape[0],dtype=torch.long).cuda()
                    ground_truthimage_text = torch.arange(image_origin.shape[0],dtype=torch.long).cuda()


                    total_text_image_loss_all=0
                    for i in range(len(iamge_textsum)):
                        total_text_image_loss=(critertion(iamge_textsum[i], ground_truthimage_text)+critertion(text_imagesum[i], ground_truthtext_image))/2 
                        total_text_image_loss_all=total_text_image_loss_all+total_text_image_loss
                    total_text_image_loss_all=total_text_image_loss_all/len(iamge_textsum)


                    classify_loss = critertion(origin_res, labels)
                    #cl_loss = critertion(l_pos_neg, cl_lables)

                    #之前是0.3  75.7 现在是0.6 73.5   0.2 74.8W
                    
                    #loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha+0.2*total_text_image_loss_all) / opt.acc_batch_size
                    #loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha+0.2*total_text_image_loss_all+3*loss_ita_hidden) / opt.acc_batch_size
                    #loss = (classify_loss +0.2*total_text_image_loss_all+2*loss_ita_hidden) / opt.acc_batch_size
                    #loss = classify_loss+0.2*total_text_image_loss_all+0.3*loss_ita_hidden
                    loss = classify_loss+0.3*loss_ita_hidden
                    #全局系数为1 adamw   74.444       ，全局系数 0(就是不用这个loss)  adamw   73.3 
                    #loss = classify_loss+0.2*total_text_image_loss_all
                    #loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha+0.3*total_text_image_loss_all) / opt.acc_batch_size
                    #loss=awl(classify_loss,cl_loss,cl_self_loss,total_text_image_loss_all)   #自动加权loss

                    loss=loss/opt.acc_grad
            #loss.backward()
            scaler.scale(loss).backward()   #混合精度
            train_loader_tqdm.set_description("Train Iteration, loss: %.6f, lr: %e" %
                                              (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/aloss_ita_hidden', loss_ita_hidden.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/atotal_text_image_loss', total_text_image_loss_all.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/classify_loss', classify_loss.item(), global_step=step_num + epoch_step_num)
                    #log_summary_writer.add_scalar('train_info/cl_loss', cl_loss.item(), global_step=step_num + epoch_step_num)
                    #log_summary_writer.add_scalar('train_info/cl_self_loss', cl_self_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/lr', optimizer.param_groups[0]['lr'], global_step=step_num + epoch_step_num)
#                    log_summary_writer.add_scalar('train_info/fuse_lr', optimizer.param_groups[1]['lr'], global_step=step_num + epoch_step_num)
                #optimizer.step()              
                scaler.step(optimizer)   #混合精度噢
                scaler.update()
                
                optimizer.zero_grad()
            step_num += 1
            
            _, predicted = torch.max(origin_res, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)
        scheduler.step()  #余热 余弦
        #scheduler.step()

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1, train_precision, train_R, run_loss)
        WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
        print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_w', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/r_w', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/p_w', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_ma', train_F1, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "train_F1_weighted": train_F1_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_R_weighted": train_R_weighted,
            "run_loss": run_loss
        }
        # debug：正常运行不要把下面的代码注释掉

        last_F1, last_Accuracy = dev_process.dev_process(opt, critertion, cl_model, dev_loader, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)
