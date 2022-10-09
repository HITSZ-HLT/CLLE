import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import M2M100Tokenizer
# from model import  M2M100ForConditionalGeneration
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration
from dataset import Translation_TestDataset
import datasets
from tqdm import tqdm
import logging
import datetime
import os
import jieba
import time
import shutil
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing

logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def bleu(model,dataloader,args):
    model.eval()
    pred_list=[]
    refs_list=[]
    tokenizer=dataloader.dataset.tokenizer
    with torch.no_grad():
        # for i,(input_ids, attention_mask, labels, tgt) in tqdm(enumerate(dataloader),f'Compute BLEU(total:{min(dataloader.dataset.__len__(),args.valid_num)})'):
        for i,(input_ids, attention_mask, labels, tgt) in enumerate(dataloader):
            if i>args.valid_num:
                break
            if i%100==0:
                logging.info(f'Compute BLEU(total:{min(dataloader.dataset.__len__(),args.valid_num)}, current:{i})')
            with torch.cuda.amp.autocast():
                model_inputs={"input_ids":input_ids.to(DEVICE),
                              "attention_mask":attention_mask.to(DEVICE)}
                gen_tokens = model.generate( **model_inputs, max_new_tokens=args.max_len,forced_bos_token_id=tgt)
                preds=tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
                if tgt == tokenizer.lang_code_to_id['zh']:
                    preds = [" ".join(jieba.cut(x)) for x in preds]
                    refs = [" ".join(jieba.cut(x)) for x in refs]

                pred_list.append(preds)
                refs_list.append(refs)
        results = METRIC.compute(predictions=pred_list, references=refs_list)

    logging.info(f'Translations --<{args.lgpair}>-- References')
    for pred, ref in zip(pred_list,refs_list):
        logging.info(f"{pred} --<{args.lgpair}>-- {ref}")

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLL_WMT Arguments', allow_abbrev=False)
    parser.add_argument('--device', type=int, default=0, help='cuda id')
    parser.add_argument('--max_len', type=int, default=80, help='max_len')
    parser.add_argument('--use_amp', type=bool, default=True, help='fp16')
    parser.add_argument('--lgpair', type=str, default='zh-en', help='tasks')
    parser.add_argument('--temp_for_test', type=str, default="../Baseline_for_LLLWMT/model_for_test", help='')
    parser.add_argument('--save_dir', type=str, default="../Baseline_for_LLLWMT/save_dir/transformer-base", help='')
    parser.add_argument('--model_file', type=str, default="Task_0-Epoch_9.bin", help='')
    parser.add_argument('--data_dir', type=str, default="../CCMatrix_lgs_zh/top1M_refined/Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir", help='data_dir')
    parser.add_argument('--sacrebleu_path', type=str, default="../transformers/download/sacrebleu.py", help='')
    parser.add_argument('--valid_num', type=int, default=10000, help='Max valid samples number(Random)')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max valid samples number(Random)')
    args = parser.parse_args()
    args.temp_for_test = args.temp_for_test +"/"+ str(time.time())
    logging.info('='*50)
    for para in args.__dict__:
        logging.info(f'{" " * (20 - len(para))}{para}={str(args.__dict__[para])}')
    logging.info('=' * 50)

    # copy model to temp dir
    if not os.path.exists(args.temp_for_test):
        os.mkdir(args.temp_for_test)

    os.system(f'cp {args.save_dir}/{args.model_file} {args.temp_for_test}/pytorch_model.bin')
    os.system(f'cp {args.save_dir}/config.json {args.temp_for_test}/config.json')

    logging.info(f'copy model and config from {args.save_dir} to {args.temp_for_test}')

    DEVICE=f'cuda:{args.device}' if args.device>-1 else "cpu"
    METRIC = datasets.load_metric(args.sacrebleu_path)
    logging.info('SacreBLRU loaded!')
    tokenizer = M2M100Tokenizer.from_pretrained(args.save_dir)
    model = M2M100ForConditionalGeneration.from_pretrained(args.temp_for_test)
    shutil.rmtree(args.temp_for_test)
    logging.info(f"Model loaded from {args.temp_for_test}")
    model.to(DEVICE)
    ## prepare dataset
    logging.info(f"Current task: {args.lgpair}")
    Dataset4dev = Translation_TestDataset(lg_pairs=[args.lgpair],
                                                     mode='test',
                                                     tokenizer=tokenizer,
                                                     data_dir=args.data_dir,
                                                     max_lenth=args.max_len,
                                                    max_samples=args.max_samples,
                                                     )
    Dev_dataloader = DataLoader(Dataset4dev, batch_size=1, shuffle=False, num_workers=1)
    logging.info("Train_dataloader prepared!")
    results = bleu(model, Dev_dataloader, args)
    logging.info(f'BLEU({args.valid_num} random samples on valid set): {results["score"]:.5}')


