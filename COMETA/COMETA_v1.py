import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import M2M100Tokenizer,M2M100Config
# from model import  M2M100ForConditionalGeneration
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration
from dataset import Multilingual_Translation_Dataset,Translation_TestDataset,Memory_Dataset
import torch.optim as optim
import datasets
import logging
import datetime
from tqdm import tqdm
import torch.optim.lr_scheduler
import os
import numpy as np
from sklearn.cluster import KMeans
import json
import  jieba
from meta_model import Loss_predict_net_dynamic as Loss_predict_net
from torch.nn import MSELoss,KLDivLoss



def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing

logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def bleu(model,dataloader,Max_test):
    model.eval()
    pred_list=[]
    refs_list=[]
    tokenizer = dataloader.dataset.tokenizer
    with torch.no_grad():

        for i,(input_ids, attention_mask, labels, tgt) in enumerate(dataloader):
            if i>Max_test:
                break
            with torch.cuda.amp.autocast():
                model_inputs={"input_ids":input_ids.to(DEVICE),
                              "attention_mask":attention_mask.to(DEVICE)}
                gen_tokens = model.generate( **model_inputs, forced_bos_token_id=tgt)
                preds=tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
                if tgt == tokenizer.lang_code_to_id['zh']:
                    preds = [" ".join(jieba.cut(x)) for x in preds]
                    refs = [" ".join(jieba.cut(x)) for x in refs]

                pred_list.append(preds)
                refs_list.append(refs)
        results = METRIC.compute(predictions=pred_list, references=refs_list)
    return results

def train_eval_one_epoch(model,memory_space,Train_dataloader,Dev_dataloader,optimizer,scaler,args,
                         meta_model,
                         meta_optimizer,
                         ):
    meta_loss_fn = MSELoss()
    def log_something():
        log_str=f"Task:{task_id}/{len(args.LLL_lgpairs) - 1}\t"+ \
                         f"Epoch:{epoch}/{args.epoches - 1}\t"+ \
                         f"Step:{i}/{N_steps_1epoch}\t"+ \
                         f"Loss:{loss.detach().cpu().item() * args.accumulation_steps:.3}\t"+ \
                         f"Loss_mt:{loss_mt.detach().cpu().item():.3}\t"+ \
                         f"Meta loss:{meta_loss.detach().cpu().item()*args.meta_accumulation_steps:.3}\t"
        if task_id>0:
            log_str = log_str + f"loss_emb:{loss_emb.detach().cpu().item():.3}\t" + \
                f"fisher_score:{memory_space.fisher_socre[idx_hty].detach().cpu().mean().item():.3}\t"
        logging.info(log_str)

    model.train()
    N_samples = Train_dataloader.dataset.__len__()
    N_steps_1epoch = int(N_samples/Train_dataloader.batch_size)
    loss_mean = 0.
    accumu_steps = 0
    for i,(input_ids, attention_mask, labels) in enumerate(Train_dataloader):
        with torch.cuda.amp.autocast():
            Seq2SeqLMOutput = model(input_ids=input_ids.to(DEVICE),
                                    attention_mask=attention_mask.to(DEVICE),
                                    labels=labels.to(DEVICE),
                                    )
            loss_mt = Seq2SeqLMOutput.loss
            if task_id != 0:
                ## retain knowledge
                ##########################################################################################
                idx_hty=memory_space.history_sight_words_list
                loss_emb = (((model.model.encoder.embed_tokens.weight[idx_hty] -
                              memory_space.words_emb[idx_hty]).square()) * memory_space.fisher_socre[idx_hty]).sum(
                    dim=1).mean()

                loss = loss_mt  + args.alpha_1*loss_emb
                ##########################################################################################
            else:
                loss = loss_mt
            loss = loss/ args.accumulation_steps
            loss_mean = loss_mean + loss_mt / args.accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % args.accumulation_steps == 0:

            ## update the meta-model by float-32
            #################################################
            input_data = get_model_feature(model, memory_space.current_sight_words).detach()
            meta_labels = loss_mean.detach().unsqueeze(dim=0).unsqueeze(dim=1).float()[0][0]
            meta_pred = meta_model(input_data.float())[0][0]

            meta_loss = meta_loss_fn(meta_pred, meta_labels)/args.meta_accumulation_steps
            meta_loss.backward()
            accumu_steps+=1
            if accumu_steps == args.meta_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
                meta_optimizer.step()
                meta_optimizer.zero_grad()
                accumu_steps = 0
            loss_mean = 0.
            #################################################

            ## update the transformer-model

            # zero non-task grad
            model.model.encoder.embed_tokens.weight.grad[memory_space.zero_grad_ids] = 0.

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            memory_space.update_fisher(model)


        if (i+1) %args.log_interval == 0:
            log_something()


        if (i + 1)%args.eval_interval ==0 and i>0:
            results = bleu(model,Dev_dataloader,args.valid_num)
            logging.info(f'BLEU({args.valid_num} random samples on valid set): {results["score"]:.5}')

def replay_eval_one_epoch(model,Train_dataloader,Dev_dataloader,optimizer,scaler,args):
    model.train()
    N_samples = Train_dataloader.dataset.__len__()
    N_steps_1epoch = int(N_samples/Train_dataloader.batch_size)
    for i,(input_ids, attention_mask, labels) in enumerate(Train_dataloader):
        if args.use_amp:
            with torch.cuda.amp.autocast():
                Seq2SeqLMOutput = model(input_ids=input_ids.to(DEVICE),
                                        attention_mask=attention_mask.to(DEVICE),
                                        labels=labels.to(DEVICE),
                                        )
                loss = Seq2SeqLMOutput.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            Seq2SeqLMOutput = model(input_ids=input_ids.to(DEVICE),
                                    attention_mask=attention_mask.to(DEVICE),
                                    labels=labels.to(DEVICE),
                                    )
            loss = Seq2SeqLMOutput.loss / args.accumulation_steps
            loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if i%args.log_interval == 0:
            logging.info(f"Task:{task_id}/{len(args.LLL_lgpairs)-1}\t"
                         f"Replay epoch:{epoch_2}/{args.epoches_replay-1}\t"
                         f"Step:{i}/{N_steps_1epoch}\t"
                         f"Loss:{loss.detach().cpu().item()*args.accumulation_steps:.5}")
        if i%args.eval_interval ==0 and i>0:
            results = bleu(model,Dev_dataloader,args.valid_num)
            logging.info(f'BLEU({args.valid_num} random samples on valid set): {results["score"]:.5}')



class Memory_space():
    def __init__(self,data_list=[],fast_sampling=20):
        # [model_inputs]
        self.data_list = data_list
        self.fast_sampling=fast_sampling
        self.sampling_history=None
        self.stop_words_ID = [0,2,3,4,5,7,9,12,17,24,26,30]
        self.words_emb = torch.zeros([cofig.vocab_size,cofig.d_model],dtype=torch.float32,device=DEVICE)
        self.fisher_socre = torch.zeros([cofig.vocab_size,cofig.d_model],dtype=torch.float32,device=DEVICE)
        # self.sight_tokenIDs=[]
        # self.zh_sight_tokenIDs = []
        self.sampling_record = {}
        self.TokenID2lgID = {}
        self.current_sight_words = []
        self.last_sight_words = []
        self.history_sight_words = []
        self.history_sight_words_list = []
        self.meta_models = []
        self.N_active_fisher = 0
        self.zero_grad_ids = []

    def update_meta_models(self,meta_model):
        self.meta_models.append(meta_model)

    def update_fisher(self,model):

        for meta_model, task_sight_words in zip(self.meta_models,self.history_sight_words):
            task_emb = get_model_feature(model, task_sight_words)
            task_fisher = 1000 * torch.autograd.grad(outputs=meta_model(task_emb)[0][0], inputs=task_emb)[0][
                0].abs()
            # logging.info(f"Debug - task_fisher : {task_fisher.detach().mean().cpu().item():.3}")
            self.fisher_socre[task_sight_words] = task_fisher

        self.fisher_socre[self.history_sight_words_list] +=1

    def update_data(self,new_list):
        # add new languages pairs
        self.data_list.extend(new_list)
        logging.info(f"Memory space updated! Total samples number: {len(self.data_list)}")


    def get_current_dataloader(self,args,tokenizer):
        Dataset_memory = Memory_Dataset(data_list=self.data_list,tokenizer=tokenizer,max_lenth=args.max_len)
        Memory_dataloader = DataLoader(Dataset_memory, batch_size=args.batch_size, shuffle=True, num_workers=4)
        return Memory_dataloader

    def update_current_sight_words(self, Dataset4train, N_sightwords, update_sightwords):
        """ lg and zh tokens """
        all_data = Dataset4train.data
        lg_pairs_num = Dataset4train.lg_pairs_num
        lg_data_idx = [sum(lg_pairs_num[:i + 1]) for i in range(len(lg_pairs_num))]
        sight_words_ = []
        zh_sight_words_ = []

        for index, lg_pair in enumerate(lg_pairs):
            e_idx = lg_data_idx[index]
            s_idx = lg_data_idx[index - 1] if index > 0 else 0

            ## statis sight-words
            lg_data = all_data[s_idx:e_idx]
            # Word frequency statistics
            tokenID_stais = {}
            zh_tokenID_stais = {}
            if lg_pair.endswith("-zh"):
                for item in lg_data:
                    for tokenID in item["input_ids"][1:]:
                        tokenID_stais[tokenID] = tokenID_stais[tokenID] + 1 if tokenID in tokenID_stais else 1
                    # # chinese tokens
                    for tokenID in item["labels"][1:]:
                        zh_tokenID_stais[tokenID] = zh_tokenID_stais[tokenID] + 1 if tokenID in zh_tokenID_stais else 1

            else:
                for item in lg_data:
                    for tokenID in item["labels"][1:]:
                        tokenID_stais[tokenID] = tokenID_stais[tokenID] + 1 if tokenID in tokenID_stais else 1
                    # # chinese tokens
                    for tokenID in item["input_ids"][1:]:
                        zh_tokenID_stais[tokenID] = zh_tokenID_stais[tokenID] + 1 if tokenID in zh_tokenID_stais else 1

            sight_words = sorted(tokenID_stais.items(), key=lambda x: x[1], reverse=True)  # 排序
            sight_words = [x[0] for x in sight_words[:N_sightwords]]
            sight_words_.extend(sight_words)

            zh_sight_words = sorted(zh_tokenID_stais.items(), key=lambda x: x[1], reverse=True)  # 排序
            zh_sight_words = [x[0] for x in zh_sight_words[:N_sightwords]]
            zh_sight_words_.extend(zh_sight_words)


        if update_sightwords:
            ## remove repeat ids
            self.current_sight_words = list(set(sight_words_ + zh_sight_words_))
            logging.info(
                f"Current sight words update, current number:{len(self.current_sight_words)}")

        self.zero_grad_ids = list(set(range(cofig.vocab_size))-set(sight_words_+zh_sight_words_))
        logging.info(f"Zero grad ids updated! number: {len(self.zero_grad_ids)}")


    def update_emb(self):
        self.words_emb[self.last_sight_words] = model.model.encoder.embed_tokens.weight[self.last_sight_words].data.to(
            DEVICE)

    def update_history(self,meta_model):
        """
        update history sightwords and words-emb
        """
        self.last_sight_words = self.current_sight_words
        self.current_sight_words = []

        self.history_sight_words.append(self.last_sight_words)

        self.history_sight_words_list.extend(self.last_sight_words)
        self.history_sight_words_list = list(set(self.history_sight_words_list))

        meta_model.eval()
        for name, param in meta_model.named_parameters():
            param.requires_grad = False
        self.meta_models.append(meta_model)
        logging.info(f"History updated : Last and history sight words, last words_emb and meta models. history_sight_words:{len(self.history_sight_words_list)}")

    def sampling_typical_samples_by_Kmeans(self,Dataset4train,model,N_samples_perlg,lg_pairs):
        """Return a list : [(input_ids,attention_mask,labels),...]"""
        all_data = Dataset4train.data
        lg_pairs_num = Dataset4train.lg_pairs_num
        lg_data_idx = [sum(lg_pairs_num[:i+1])  for i in range(len(lg_pairs_num))]
        sampled_index_=[]
        sampled_data_=[]

        model.eval()
        for index, lg_pair in enumerate(lg_pairs):
            e_idx = lg_data_idx[index]
            s_idx = lg_data_idx[index-1] if index>0 else 0

            ## statis sight-words
            lg_data = all_data[s_idx:e_idx]


            # Only select lg_pair data
            Dataset4train.data = lg_data
            sampling_dataloader = DataLoader(Dataset4train, batch_size=args.batch_size, shuffle=False, num_workers=4)

            features = []
            with torch.no_grad():
                for i, (input_ids, attention_mask, labels) in enumerate(sampling_dataloader):
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            BaseModelOutput = model.model.encoder(input_ids=input_ids.to(DEVICE),
                                                    attention_mask=attention_mask.to(DEVICE),
                                                    output_hidden_states=True)
                            # (B, sequence_length, hidden_size)
                            last_hidden_state = BaseModelOutput.last_hidden_state
                            # (B, hidden_size)
                            for k in range(last_hidden_state.shape[0]):
                                length=attention_mask[k,:].sum()
                                feature = last_hidden_state[k,:length,:].mean(dim=0).detach().cpu().numpy()
                                features.append(feature)
            logging.info(f"{lg_pair} feature embedding getted!")
            features = np.stack(features)
            distances = KMeans(n_clusters=self.fast_sampling, random_state=0).fit_transform(features)
            N_per_cls = int(N_samples_perlg/self.fast_sampling)
            sampled_index = []
            for k in range(self.fast_sampling):
                sel_index = np.argpartition(distances[:, k], N_per_cls)[:N_per_cls].tolist()
                sampled_index.extend(sel_index)
            # Remove Duplicates
            # Index according to
            sampled_index = list(set(sampled_index))
            sampled_data = [Dataset4train.data[index] for index in sampled_index]
            sampled_data_.extend(sampled_data)
            sampled_index_.extend([s_idx+idx for idx in sampled_index] )
            logging.info(f"{lg_pair} sampling finished, total: {lg_pairs_num[index]}, sampling:{len(sampled_data)}")
            # Clear
            del sampling_dataloader,sampled_data,sampled_index,features,distances
        del Dataset4train
        return sampled_data_,sampled_index_,


def Save_training(args, task_id, epoch, model, tokenizer,memory_space,meta_model=None):
    torch.save(model.state_dict(), f"{args.save_dir}/Task_{task_id}-Epoch_{epoch}.bin")
    model.config.to_json_file(f"{args.save_dir}/config.json")
    tokenizer.save_pretrained(f"{args.save_dir}")

    with open(f'{args.save_dir}/record.txt', 'w', encoding='utf-8') as w:
        w.write(f"Task\tEpoch\n{task_id}\t{epoch}")
    logging.info(f"Model saved to: {args.save_dir}/Task_{task_id}-epoch_{epoch}.bin")

    if meta_model:
        torch.save(meta_model.state_dict(), f"{args.save_dir}/Task_{task_id}-Meta_model.bin")
        logging.info(f"Meta model saved to: {args.save_dir}/Task_{task_id}-Meta_model.bin")

    torch.save(memory_space.current_sight_words,f"{args.save_dir}/Task_{task_id}-Sightwords.pt")
    logging.info(f"Sightwords ID saved to: {args.save_dir}/Task_{task_id}-Sightwords.pt")

    torch.save(memory_space.words_emb.cpu(), f"{args.save_dir}/Task_{task_id}-memory_emb.pt")
    logging.info(f"Memory words embedding saved to: {args.save_dir}/Task_{task_id}-memory_emb.pt")




def get_model_feature(model,sight_ID=None):

    return model.model.encoder.embed_tokens.weight[sight_ID].unsqueeze(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLL_WMT Arguments', allow_abbrev=False)
    parser.add_argument('--device', type=int, default=-1, help='cuda id')
    parser.add_argument('--max_len', type=int, default=80, help='max_len')
    parser.add_argument('--epoches', type=int, default=1, help='N_epoches')
    parser.add_argument('--epoches_replay', type=int, default=1, help='N_epoches')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation_steps')
    parser.add_argument('--meta_accumulation_steps', type=int, default=1, help='*meta_accumulation_steps')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--use_amp', type=bool, default=True, help='fp16')
    parser.add_argument('--LLL_lgpairs', type=str, default="[['zh-en','en-zh'],['zh-nl','nl-zh'],['zh-fr','fr-zh']]", help='tasks')

    parser.add_argument('--lr', type=float, default=5e-4, help='lr')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='lr')
    parser.add_argument('--alpha_1', type=float,  default=20.0, help='Loss weight for embedding part.')
    parser.add_argument('--alpha_2', type=float, default=20.0, help='Loss weight for meta part.')
    parser.add_argument('--chk_dir', type=str, default="../transformers/download", help='')
    parser.add_argument('--Finetuning', action='store_true', help='')
    parser.add_argument('--sacrebleu_path', type=str, default="../transformers/download/sacrebleu.py", help='')
    parser.add_argument('--data_dir', type=str, default="../CCMatrix_lgs_zh/top1M_refined/Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir", help='data_dir')
    parser.add_argument('--save_dir', type=str, default="../Baseline_for_LLLWMT/save_dir/LLL_baseline_debug", help='save dir')
    parser.add_argument('--config_dir', type=str, default="../Baseline_for_LLLWMT/config_dir", help='config_dir')
    parser.add_argument('--model_name', type=str, default="transformer-small",choices=['M2M-418M','transformer-small','transformer-base','transformer-big'], help='')
    parser.add_argument('--log_interval', type=int, default=8, help='steps')
    parser.add_argument('--eval_interval', type=int, default=5000, help='steps')
    parser.add_argument('--valid_num', type=int, default=100, help='Max valid samples number(Random)')
    parser.add_argument('--Memory_size', type=int, default=1000, help='Storing examples per language')
    parser.add_argument('--N_sightwords', type=int, default=5000, help='Nunber of sightwords per language')
    parser.add_argument('--max_train_samples', type=int, default=10000, help='max training samples')
    parser.add_argument('--Replay', action='store_true', help='wether use replay')

    args = parser.parse_args()
    args.LLL_lgpairs = eval(args.LLL_lgpairs)
    logging.info('='*50)
    for para in args.__dict__:
        logging.info(f'{" " * (20 - len(para))}{para}={str(args.__dict__[para])}')

    logging.info('=' * 50)
    DEVICE=f'cuda:{args.device}' if args.device>-1 else "cpu"

    # sacrebleu
    METRIC = datasets.load_metric(args.sacrebleu_path)
    # METRIC = None
    logging.info('SacreBLEU loaded!')



    # load tokenizer and model
    tokenizer = M2M100Tokenizer.from_pretrained(args.chk_dir)
    logging.info(f"Tokenizer from {args.chk_dir}")

    cofig = M2M100Config.from_json_file(f'{args.config_dir}/{args.model_name}.json')
    if args.Finetuning:
        model = M2M100ForConditionalGeneration.from_pretrained(args.chk_dir)
        logging.info(f"M2M-100 loaded from {args.chk_dir}")
    else:
        model = M2M100ForConditionalGeneration(cofig)
        logging.info(f"Model <{args.model_name}> random inited!")

    # Init Memory space
    memory_space = Memory_space()
    logging.info("Memory space init!")
    sampling_index_record = {}



    logging.info(f'Model parameters number: {model.num_parameters()}')
    model.to(DEVICE)
    model.train()


    # Incremental languages
    for task_id,lg_pairs in enumerate(args.LLL_lgpairs):


        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None


        # Train a new meta model for current task loss prediction
        meta_model = Loss_predict_net(d_model=cofig.d_model).to(DEVICE)
        meta_model.train()
        meta_optimizer = optim.Adam([
            {'params': meta_model.parameters(), 'lr': args.meta_lr},
        ])

        if task_id == 0:
            logging.info(meta_model)
            logging.info(f'Meta-model parameters number: {meta_model.num_parameters()}')


        # prepare train and vaild dataset&dataloader
        logging.info(f"Current task: {','.join(lg_pairs)}")
        Dataset4train = Multilingual_Translation_Dataset(lg_pairs=lg_pairs,
                                                         mode='train',## for debug
                                                         tokenizer=tokenizer,
                                                         data_dir=args.data_dir,
                                                         max_lenth=args.max_len,
                                                         max_samples=args.max_train_samples,
                                                         )
        Train_dataloader = DataLoader(Dataset4train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=4)
        logging.info(f"Train_dataloader prepared! Data number: {Dataset4train.__len__()}")


        Dataset4dev = Translation_TestDataset(lg_pairs=lg_pairs,
                                              mode='valid',
                                              tokenizer=tokenizer,
                                              data_dir=args.data_dir,
                                              max_lenth=args.max_len,
                                              )

        Dev_dataloader = DataLoader(Dataset4dev, batch_size=1, shuffle=True, num_workers=1)
        logging.info(f"Valid_dataloader prepared! Data number: {Dataset4dev.__len__()}")

        memory_space.update_current_sight_words(Dataset4train,args.N_sightwords,True)
        # Current task training

        if task_id==0:
            optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr},
            ])

        for epoch in range(args.epoches):
            if task_id>0:
                if epoch==0:
                    ## 1-st epoch freeze encoder-decoder
                    optimizer = torch.optim.Adam(
                        [{"params": [para for name, para in model.named_parameters() if "shared" in name]}],
                        lr=args.lr)
                if epoch==1:
                    optimizer = optim.Adam([
                        {'params': model.parameters(), 'lr': args.lr},
                    ])


            train_eval_one_epoch(model,
                                 memory_space,
                                Train_dataloader,
                                Dev_dataloader,
                                optimizer,
                                scaler,
                                args,
                                meta_model,
                                meta_optimizer,
                                 )

            new_N_sightwords = int((args.N_sightwords)*(2.5**(epoch+1)))
            memory_space.update_current_sight_words(Dataset4train, new_N_sightwords, False)

        # update sightwords record
        memory_space.update_history(meta_model)
        # update emb record
        memory_space.update_emb()

        Save_training(args, task_id, epoch, model, tokenizer, memory_space, meta_model)

        if not args.Replay:
            continue


        ## <Memory Replay>
        if str(task_id) not in sampling_index_record:
            sampled_data, sampled_index = memory_space.sampling_typical_samples_by_Kmeans(Dataset4train,
                                                                                          model,
                                                                                          args.Memory_size,
                                                                                          lg_pairs)

            logging.info(f"Memory sampling finished! Number: {len(sampled_data)}")
            sampling_index_record[str(task_id)] = str(sampled_index)
            # Record the sampling_index_record
            with open(f"{args.save_dir}/sampling_index_record.json","w",encoding="utf-8") as w:
                json.dump(sampling_index_record,w)
                logging.info(f"Update {args.save_dir}/sampling_index_record.json")

            # Step2: Update memory space
            memory_space.update_data(sampled_data)
        else:
            logging.info("Skip memory replay step-1 and step-2")

        # task-0 does not replay
        if task_id ==0:
            Save_training(args, task_id, 999, model, tokenizer, memory_space)
            continue

        # Step3: Get Memory_dataloader
        Memory_dataloader=memory_space.get_current_dataloader(args,tokenizer)
        logging.info('Memory_dataloader prepared!')
        for epoch_2 in range(args.epoches_replay):
            replay_eval_one_epoch(model, Memory_dataloader, Dev_dataloader, optimizer, scaler, args)

        Save_training(args, task_id, 999, model, tokenizer, memory_space)

        # after replay update emb record
        memory_space.update_emb()