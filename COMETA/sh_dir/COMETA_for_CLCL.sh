#!/bin/bash
Time=$(date "+%Y%m%d-%H%M%S")
cd /userhome/Research_HUB/LLL_WMT/Baseline_for_LLLWMT
base_dir=/userhome/Research_HUB/LLL_WMT
Data_dir=${base_dir}/CCMatrix_lgs_zh/top1M_refined/Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir

v='1'
exp='1'

save_dir=${base_dir}/Baseline_for_LLLWMT/save_dir/LLL_baseline_v${v}_exp${exp}
if [ ! -d ${save_dir}  ];then
  mkdir ${save_dir}
fi

python COMETA_v${v}.py --device $1 \
--chk_dir ${base_dir}/transformers/download \
--config_dir ${base_dir}/Baseline_for_LLLWMT/config_dir \
--sacrebleu_path ${base_dir}/transformers/download/sacrebleu.py \
--data_dir ${Data_dir} \
--max_len 80 \
--accumulation_steps 8 \
--batch_size 128 \
--lr 5e-4 \
--model_name transformer-small \
--log_interval 20 \
--eval_interval 5000 \
--LLL_lgpairs "[['zh-de','de-zh','zh-nl','nl-zh','zh-sv','sv-zh','zh-pt','pt-zh','zh-es','es-zh','zh-ru','ru-zh','zh-cs','cs-zh'],['zh-en','en-zh','zh-fr','fr-zh','zh-pl','pl-zh']]"  \
--save_dir ${save_dir} \
--valid_num 100 \
--epoches 3 \
--epoches_replay 3 \
--N_sightwords 3000 \
--alpha_1 20 \
--Memory_size 2000 \
--max_train_samples 1000000 \
--alpha_2 0.1 \
--meta_lr 1e-3 \
--Replay \
--meta_accumulation_steps 16  >log_dir/LLLNMT_baseline_10lgs_v${v}_exp${exp}_${Time}.log 2>&1