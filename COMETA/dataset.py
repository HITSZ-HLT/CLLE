import torch
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
import random

class Translation_TestDataset(Dataset):
    """ ramdom 300 samples"""
    def __init__(self,
                 lg_pairs=[],
                 data_dir='/userhome/LLL_WMT/CCMatrix_lgs_zh/top1M_refined/'
                          'Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir',
                 mode='test',
                 tokenizer=None,
                 max_lenth=128,
                 max_samples=1e8,
                 ):
        self.max_lenth=max_lenth
        assert mode in ['valid','test']
        self.data=[]
        self.tokenizer = tokenizer
        random.seed(20220330)
        for lg_pair in lg_pairs:
            src,tgt = lg_pair.split("-")
            if src == 'zh':
                lg_pair = f"{tgt}-{src}"
            tokenizer.src_lang = src
            tokenizer.tgt_lang = tgt
            with open(f"{data_dir}/spm.{mode}.{lg_pair}.{src}",encoding="utf-8") as f1:
                with open(f"{data_dir}/spm.{mode}.{lg_pair}.{tgt}",encoding="utf-8") as f2:
                    lines_1 = f1.readlines()
                    lines_2 = f2.readlines()
                    Index = list(range(len(lines_1)))
                    random.shuffle(Index)

                    # for i,(src_txt,tgt_txt) in tqdm(enumerate(zip(lines_1[Index],lines_2[Index])),f"load spm.{src}-{tgt}"):
                    for i,idx in enumerate(Index):
                        src_txt=lines_1[Index[idx]]
                        tgt_txt=lines_2[Index[idx]]
                        src_tokens=[tokenizer.lang_code_to_token[src]]+src_txt.strip().split()
                        tgt_tokens=[tokenizer.lang_code_to_token[tgt]]+tgt_txt.strip().split()
                        src_tokens = src_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        tgt_tokens = tgt_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        model_inputs = {}
                        model_inputs['input_ids'] = tokenizer.convert_tokens_to_ids(src_tokens)
                        model_inputs["labels"] = tokenizer.convert_tokens_to_ids(tgt_tokens)
                        model_inputs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt)
                        self.data.append(model_inputs)
                        if i>=max_samples-1:
                            break


    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids']
        attention_mask = [1]*(len(model_inputs['input_ids']))
        labels = model_inputs['labels']

        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(labels),model_inputs["forced_bos_token_id"]

    def __len__(self,):
        return len(self.data)


class Multilingual_Translation_Dataset(Dataset):
    def __init__(self,
                 lg_pairs=[],
                 data_dir='/userhome/LLL_WMT/CCMatrix_lgs_zh/top1M_refined/'
                          'Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir',
                 mode='test',
                 tokenizer=None,
                 max_lenth=128,
                 max_samples=1e8,
                 ):
        self.max_lenth=max_lenth
        assert mode in ['train','valid','test']
        self.data=[]
        self.tokenizer = tokenizer
        self.lg_pairs_num=[]
        for lg_pair in lg_pairs:
            src,tgt = lg_pair.split("-")
            if src == 'zh':
                lg_pair = f"{tgt}-{src}"
            tokenizer.src_lang = src
            tokenizer.tgt_lang = tgt
            with open(f"{data_dir}/spm.{mode}.{lg_pair}.{src}",encoding="utf-8") as f1:
                with open(f"{data_dir}/spm.{mode}.{lg_pair}.{tgt}",encoding="utf-8") as f2:
                    for idx,(src_txt,tgt_txt) in enumerate(zip(f1,f2)):
                        src_tokens=[tokenizer.lang_code_to_token[src]]+src_txt.strip().split()
                        tgt_tokens=[tokenizer.lang_code_to_token[tgt]]+tgt_txt.strip().split()
                        src_tokens = src_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        tgt_tokens = tgt_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        model_inputs = {}
                        model_inputs['input_ids'] = tokenizer.convert_tokens_to_ids(src_tokens)
                        model_inputs["labels"] = tokenizer.convert_tokens_to_ids(tgt_tokens)
                        self.data.append(model_inputs)
                        if idx >=max_samples-1:# debug
                            break
            self.lg_pairs_num.append(idx+1)

    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids'] + [self.tokenizer.pad_token_id] * (
                    self.max_lenth - len(model_inputs['input_ids']))
        attention_mask = [1]*(len(model_inputs['input_ids'])) + [0] * (self.max_lenth - len(model_inputs['input_ids']))
        labels = model_inputs['labels'] + [-100] * (self.max_lenth - len(model_inputs['labels']))
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(labels)
    def __len__(self,):
        return len(self.data)


class Multilingual_Index_Dataset(Dataset):
    def __init__(self,
                 lg_pairs=[],
                 data_dir='/userhome/LLL_WMT/CCMatrix_lgs_zh/top1M_refined/'
                          'Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir',
                 feature_dir='/userhome/LLL_WMT/CCMatrix_lgs_zh/top1M_refined/'
                          'Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/feature_dir',
                 mode='train',
                 tokenizer=None,
                 max_lenth=128,
                 max_samples=1e8,
                 ):
        self.max_lenth=max_lenth
        assert mode in ['train']
        self.data=[]
        self.tokenizer = tokenizer
        self.lg_pairs_num=[]
        self.lg_pairs = lg_pairs
        self.features = []
        for lg_pair in lg_pairs:
            src,tgt = lg_pair.split("-")
            if src == 'zh':
                lg_pair = f"{tgt}-{src}"
            tokenizer.src_lang = src
            tokenizer.tgt_lang = tgt
            with open(f"{data_dir}/spm.{mode}.{lg_pair}.{src}",encoding="utf-8") as f1:
                with open(f"{data_dir}/spm.{mode}.{lg_pair}.{tgt}",encoding="utf-8") as f2:
                    for idx,(src_txt,tgt_txt) in enumerate(zip(f1,f2)):
                        src_tokens=[tokenizer.lang_code_to_token[src]]+src_txt.strip().split()
                        tgt_tokens=[tokenizer.lang_code_to_token[tgt]]+tgt_txt.strip().split()
                        src_tokens = src_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        tgt_tokens = tgt_tokens[:self.max_lenth - 1] + [tokenizer.eos_token]
                        model_inputs = {}
                        model_inputs['input_ids'] = tokenizer.convert_tokens_to_ids(src_tokens)
                        model_inputs["labels"] = tokenizer.convert_tokens_to_ids(tgt_tokens)
                        self.data.append(model_inputs)
                        if idx >=max_samples-1:# debug
                            break

            feature = torch.load(f"{feature_dir}/feature.train.{lg_pair}.{src}")[:idx+1]
            self.features.append(feature)
            self.lg_pairs_num.append(idx+1)
            assert self.features[-1].shape[0] == self.lg_pairs_num[
                -1], f"features number:{self.features[-1].shape[0]}\t data number:{self.lg_pairs_num[-1]}"
        self.features = torch.cat(self.features)

    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids'] + [self.tokenizer.pad_token_id] * (
                    self.max_lenth - len(model_inputs['input_ids']))
        attention_mask = [1]*(len(model_inputs['input_ids'])) + [0] * (self.max_lenth - len(model_inputs['input_ids']))
        labels = model_inputs['labels'] + [-100] * (self.max_lenth - len(model_inputs['labels']))
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(labels),self.features[index,:]
    def __len__(self,):
        return len(self.data)




class Memory_Index_Dataset(Dataset):
    def __init__(self,
                 data_list=None,
                 tokenizer=None,
                 max_lenth=128,
                 features=None,
                 ):
        """ datalist [model_inputs]"""
        self.data=data_list
        self.tokenizer = tokenizer
        self.max_lenth = max_lenth
        self.features = features
    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids'] + [self.tokenizer.pad_token_id] * (
                    self.max_lenth - len(model_inputs['input_ids']))
        attention_mask = [1]*(len(model_inputs['input_ids'])) + [0] * (self.max_lenth - len(model_inputs['input_ids']))
        labels = model_inputs['labels'] + [-100] * (self.max_lenth - len(model_inputs['labels']))
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(labels),self.features[index,:]
    def __len__(self,):
        return len(self.data)

class Memory_Dataset(Dataset):
    def __init__(self,
                 data_list=None,
                 tokenizer=None,
                 max_lenth=128,
                 ):
        """ datalist [model_inputs]"""
        self.data=data_list
        self.tokenizer = tokenizer
        self.max_lenth = max_lenth
    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids'] + [self.tokenizer.pad_token_id] * (
                    self.max_lenth - len(model_inputs['input_ids']))
        attention_mask = [1]*(len(model_inputs['input_ids'])) + [0] * (self.max_lenth - len(model_inputs['input_ids']))
        labels = model_inputs['labels'] + [-100] * (self.max_lenth - len(model_inputs['labels']))
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(labels)
    def __len__(self,):
        return len(self.data)

class Memory_Dataset_v5(Dataset):
    def __init__(self,
                 data_list=None,
                 feature_list=None,
                 tokenizer=None,
                 max_lenth=128,
                 ):
        """ datalist [model_inputs]"""
        assert len(data_list)==len(feature_list)
        self.data=data_list
        self.feature=feature_list
        self.tokenizer = tokenizer
        self.max_lenth = max_lenth
    def __getitem__(self, index):
        model_inputs=self.data[index]
        input_ids = model_inputs['input_ids'] + [self.tokenizer.pad_token_id] * (
                    self.max_lenth - len(model_inputs['input_ids']))
        attention_mask = [1]*(len(model_inputs['input_ids'])) + [0] * (self.max_lenth - len(model_inputs['input_ids']))
        labels = model_inputs['labels'] + [-100] * (self.max_lenth - len(model_inputs['labels']))

        return torch.LongTensor(input_ids),\
               torch.LongTensor(attention_mask),\
               torch.LongTensor(labels),\
               torch.tensor(self.feature[index],dtype=torch.float32)

    def __len__(self,):
        return len(self.data)


if __name__ == '__main__':
    from transformers import M2M100Tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained("../transformers/download")

    Dataset4train = Multilingual_Translation_Dataset(lg_pairs=['zh-en'],
                                                       mode='train',
                                                       tokenizer=tokenizer,
                                                       data_dir='../CCMatrix_lgs_zh/top1M_refined/'
                                                                'Add_Topic_Index/filter_by_TopicNum_Mediam/train_valid_test/spm_dir',
                                               )
    print(len(Dataset4train.data))
