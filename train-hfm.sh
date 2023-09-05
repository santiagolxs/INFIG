
lr=5e-6
fuse_lr=2e-5
image_output_type=all
optim=adam
data_type=HFM
train_fuse_model_epoch=20
epoch=100
warmup_step_epoch=2
text_model=debertav3_base   #'debertav3_base'  'bert-base'
fuse_type=att
batch=16
acc_grad=1
tran_num_layers=3
image_num_layers=6
queue_size=3584            #single is 3584-32-1 multiple is 13568-32-1 hfm is 19968-32-acgrad4
torchrun --nproc_per_node=4 main.py -cuda -gpu_num 0,1,2,3 -epoch ${epoch} -num_workers 8 \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model resnet-50 \
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type}\
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch} -queue_size ${queue_size}
