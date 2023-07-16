import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, TensorDataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer, BartForCausalLM
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
# from models import SpanPredictor, SchemePredictor, SpanSchemePredictor, SpanSchemePipelinedPredictor, ArgGenerator, ArgGeneratorMultiDecoderHF
from models_v2 import FactSpanIdentifier_M1, SpanSchemePredictor_M2, SpanSchemePipelinedPredictor_M2, \
    DecoderOnlyArgGenerator_M3, EncoderDecoderArgGenerator_M3#, EncoderDecoderArgGenerator_V2_M3
import nltk
from nltk.translate.bleu_score import corpus_bleu
import copy
import os
import torch.nn as nn
import mappings
from difflib import SequenceMatcher
import re

nltk.download('punkt')

"""
Branch main: 
    (i) Logic to Text Generator
    (ii) Text to Logic Generator
Branch logic_from_building_blocks:
    (i)Text to Logic building blocks using interaction layer
scheme_span_baselines: Predict the arg scheme with walton's broad classes + knowledge span prediction
"""


class Trainer:
    def __init__(self, configuration):
        self.set_random_seeds(random_seed=42)
        self.configuration = configuration

        if self.configuration["model_type"].startswith("arg_gen_M3"):
            self.special_tokens_dict = {'additional_special_tokens': sorted(list(mappings.generator_token_M3.values()))}
        else:
            self.special_tokens_dict = None

        self.device = torch.device(
            "cuda:{}".format(configuration["device_num"])) if torch.cuda.is_available() else "cpu"
        print("Device: ", self.device)
        self.tokenizer, self.model = self.get_model()
        if self.configuration["train_distributed"]:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                       device_ids=[configuration["device_num"]],
                                                                       output_device=configuration["device_num"],
                                                                       find_unused_parameters=configuration[
                                                                           "find_unused_parameters"])
        else:
            self.ddp_model = self.model
        print("DDP Model loaded")
        print(f'The model has {self.count_parameters(self.ddp_model):,} trainable parameters')

        if self.configuration["cross_validation"] and self.configuration.get("model_name", None) is not None and\
                self.configuration["device_num"] == 0:
            torch.save(self.ddp_model.state_dict(), self.configuration["model_name"])

        if not self.configuration["experiment_details"].get("use_pc", True) and \
                self.configuration["model_type"] == "arg_gen_M3_V2" and \
                self.configuration["device_num"] == 0:
            torch.save(self.ddp_model.state_dict(), self.configuration["temp_model_name"])

        if self.configuration["train_distributed"]:
            torch.distributed.barrier()
        self.data_dict = pickle.load(open(self.configuration["experiment_details"]["data_file"], "rb"))

        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.ddp_model.parameters(), lr=self.configuration["lr"])

    def pad_sequence(self, pad, batch, left=False):
        maxlen = max([len(i) for i in batch])
        lst = []
        for i in batch:
            if left:
                lst.append([pad] * (maxlen - len(i)) + i)
            else:
                lst.append(i + [pad] * (maxlen - len(i)))
        return lst

    def pad_sequence_3d(self, pad, batch):
        max_len = max([len(j) for i in batch for j in i])
        max_splits = max([len(i) for i in batch])
        padded_lst = []
        for i in batch:
            if type(i) != list:
                i = i.tolist()
            tmp_lst = [j + [pad] * (max_len - len(j)) for j in i]
            tmp_lst.extend([[pad] * max_len] * (max_splits - len(tmp_lst)))
            padded_lst.append(tmp_lst)
        return padded_lst

    def get_example(self, ky_lst):
        txt_lst, lbl_idm_red_lst, lbl_bio = [], [], []
        for i in ky_lst:
            txt_lst.append(self.data_dict[i]["model_2_format"]["text_input_ids"])
            lbl_idm_red_lst.append(self.data_dict[i]["model_2_format"]["lbl_idiom_new"])
            lbl_bio.append(self.data_dict[i]["model_2_format"]["bio_labels"].tolist())

        txt_lst = self.pad_sequence(self.tokenizer.pad_token_id, txt_lst)
        lbl_bio = self.pad_sequence(-1, lbl_bio)

        return torch.tensor(txt_lst).long(), torch.tensor(lbl_idm_red_lst).float(), torch.tensor(lbl_bio).long()

    def get_gen_example(self, ky_lst):
        encoder1_input_id_lst, encoder2_input_id_lst, encoder1_attn_lst, encoder2_attn_lst = [], [], [], []
        decoder1_input_id_lst, decoder1_labels_lst, decoder2_input_id_lst, decoder2_labels_lst = [], [], [], []
        decoder1_attn_lst, decoder2_attn_lst = [], []

        ds = "gpt2" if "gpt2" in self.configuration["experiment_details"]["base_model"] else "bart"
        left = True if "gpt2" in self.configuration["experiment_details"]["base_model"] else False

        for idxx, ix in enumerate(ky_lst):
            input_ids, out_dict, ctrl_dict = self.data_dict[ix]["model_3_format"][ds]["in_dict"]["input_ids"], \
                                             self.data_dict[ix]["model_3_format"][ds]["out_dict"], \
                                             self.data_dict[ix]["model_3_format"][ds]["ctrl"]
            ctrl_ids = ctrl_dict[self.configuration["experiment_details"]["control"] + "_ids"]

            if ds == "gpt2":
                """ 
                Our input to GPT2: Topic<VAR_0>Text...<VAR_1>Text...<EOS><CTRL><PAT>Text..<ARG>Text...<EOS>  
                Our label to GPT2: [-100..................................-100 -100 Text..<ARG>Text...<EOS>]
                
                gpt2 shifted inpt: Topic<VAR_0>Text...<VAR_1>Text...<EOS><CTRL><PAT>Text..<ARG>Text... 
                gpt2 shifted labl: [................................-100  -100  Text..<ARG>Text...<EOS>]
                """
                # if self.configuration["experiment_details"]["target"] == "pattern_argument":
                # decoder1_input_ids = input_ids + ctrl_ids + out_dict["pattern_input_ids"] + out_dict["argument_input_ids"]
                tgt_ky = self.configuration["experiment_details"]["target"] + "_input_ids"
                encoder1_input_ids, encoder2_input_ids, decoder2_input_ids, decoder2_labels = None, None, None, None
                encoder1_attn_mask, encoder2_attn_mask, decoder2_attn_mask = None, None, None
                decoder1_input_ids = input_ids + ctrl_ids + out_dict[tgt_ky]
                decoder1_attn_mask = [1] * len(decoder1_input_ids)
                if self.configuration["experiment_details"]["target"] == "pattern_argument":
                    pat_eos_loc = [p_ix for p_ix, p_id in enumerate(out_dict[tgt_ky])
                                   if p_id == self.tokenizer.eos_token_id][0] + 1
                    decoder1_labels = [-100] * (len(input_ids + ctrl_ids) + 1) + out_dict[tgt_ky][1:pat_eos_loc] + [
                        -100] + out_dict[tgt_ky][pat_eos_loc + 1:]
                else:
                    decoder1_labels = [-100] * (len(input_ids + ctrl_ids) + 1) + out_dict[tgt_ky][1:]

            else:
                if self.configuration["experiment_details"]["type"] in ["arg_gen_M3_V2", "arg_gen_M3_V3"]:
                    if idxx == 0:
                        print("\nThis is the V2 experiments with pipelined architecture, reusing the same model!\n")
                    exp_id = self.configuration["experiment_details"].get("alt_id", self.configuration["experiment_details"]["exp_id"])
                    d_n = self.data_dict[ix]["model_3_format"][ds]["V2_" + str(exp_id)]
                    if not self.configuration["experiment_details"]["multi_encoder"]:
                        encoder2_input_ids, encoder2_attn_mask, decoder2_input_ids, \
                        decoder2_attn_mask, decoder2_labels = None, None, None, None, None
                        encoder1_input_ids, encoder1_attn_mask = d_n["arg_encoder_input_ids"], \
                                                                 [1] * len(d_n["arg_encoder_input_ids"])
                        decoder1_input_ids, decoder1_attn_mask = d_n["arg_decoder_input_ids"], \
                                                                 [1] * len(d_n["arg_decoder_input_ids"])
                        decoder1_labels = d_n["arg_decoder_labels"]
                    else:
                        encoder1_input_ids, encoder1_attn_mask = d_n["pat_encoder_input_ids"], \
                                                                 [1] * len(d_n["pat_encoder_input_ids"])
                        decoder1_input_ids, decoder1_attn_mask = d_n["pat_decoder_input_ids"], \
                                                                 [1] * len(d_n["pat_decoder_input_ids"])
                        decoder1_labels = d_n["pat_decoder_labels"]

                        encoder2_input_ids, encoder2_attn_mask = d_n["arg_encoder_input_ids"], \
                                                                 [1] * len(d_n["arg_encoder_input_ids"])
                        decoder2_input_ids, decoder2_attn_mask = d_n["arg_decoder_input_ids"], \
                                                                 [1] * len(d_n["arg_decoder_input_ids"])
                        decoder2_labels = d_n["arg_decoder_labels"]

                else:
                    """
                    Encoder 1 inp: <BOS>Topic<VAR_0>Text...<VAR_1>Text...<EOS>
                    Decoder 1 inp: <BOS><CTRL><PAT>Text.......
                    Decoder 1 lbl: [-100 -100 Text..........<EOS>]
                    
                    Encoder 2 inp: <BOS>Topic<VAR_0>Text...<VAR_1>Text...<EOS><BOS><CTRL><PAT>Text....<EOS>
                    Decoder 2 inp: <BOS><CTRL><ARG>Text.......
                    Decoder 2 lbl: [-100 -100 Text..........<EOS>]
                    """
                    encoder1_input_ids = input_ids
                    encoder1_attn_mask = [1] * len(encoder1_input_ids)
                    decoder1_pre_input = out_dict["pattern_input_ids"][:1] + ctrl_ids + out_dict["pattern_input_ids"][1:]
                    decoder1_input_ids = decoder1_pre_input[:-1]
                    decoder1_attn_mask = [1] * len(decoder1_input_ids)
                    decoder1_labels = [-100] * (len(ctrl_ids) + 1) + out_dict["pattern_input_ids"][2:]

                    encoder2_input_ids = encoder1_input_ids + decoder1_pre_input
                    encoder2_attn_mask = [1] * len(encoder2_input_ids)
                    decoder2_input_ids = out_dict["argument_input_ids"][:1] + ctrl_ids + out_dict["argument_input_ids"][
                                                                                         1:-1]
                    decoder2_attn_mask = [1] * len(decoder2_input_ids)
                    decoder2_labels = [-100] * (len(ctrl_ids) + 1) + out_dict["argument_input_ids"][2:]

            decoder1_input_id_lst.append(decoder1_input_ids)
            decoder1_attn_lst.append(decoder1_attn_mask)
            decoder1_labels_lst.append(decoder1_labels)
            # if ds != "gpt2":
            if encoder2_input_ids is not None:
                encoder2_input_id_lst.append(encoder2_input_ids)
            if encoder2_attn_mask is not None:
                encoder2_attn_lst.append(encoder2_attn_mask)
            if decoder2_input_ids is not None:
                decoder2_input_id_lst.append(decoder2_input_ids)
            if decoder2_attn_mask is not None:
                decoder2_attn_lst.append(decoder2_attn_mask)
            if decoder2_labels is not None:
                decoder2_labels_lst.append(decoder2_labels)
            if encoder1_input_ids is not None:
                encoder1_input_id_lst.append(encoder1_input_ids)
            if encoder1_attn_mask is not None:
                encoder1_attn_lst.append(encoder1_attn_mask)

        decoder1_input_mat = torch.tensor(
            self.pad_sequence(self.tokenizer.pad_token_id, decoder1_input_id_lst, left=left)).long()
        decoder1_attn_mat = torch.tensor(self.pad_sequence(0, decoder1_attn_lst, left=left)).long()
        decoder1_labels_mat = torch.tensor(self.pad_sequence(-100, decoder1_labels_lst, left=left)).long()
        assert decoder1_input_mat.shape == decoder1_attn_mat.shape == decoder1_labels_mat.shape
        dct1 = {"decoder1_input_mat": decoder1_input_mat, "decoder1_attn_mat": decoder1_attn_mat,
                "decoder1_labels_mat": decoder1_labels_mat}
        dct2 = {}
        # if ds != "gpt2":
        if len(encoder1_input_id_lst) > 0 and len(encoder1_attn_lst) > 0:
            encoder1_input_mat = torch.tensor(
                self.pad_sequence(self.tokenizer.pad_token_id, encoder1_input_id_lst, left=left)).long()
            encoder1_attn_mat = torch.tensor(self.pad_sequence(0, encoder1_attn_lst, left=left)).long()
            assert encoder1_input_mat.shape == encoder1_attn_mat.shape
            dct2["encoder1_input_mat"], dct2["encoder1_attn_mat"] = encoder1_input_mat, encoder1_attn_mat

        if len(encoder2_input_id_lst) > 0 and len(encoder2_attn_lst) > 0:
            encoder2_input_mat = torch.tensor(
                self.pad_sequence(self.tokenizer.pad_token_id, encoder2_input_id_lst, left=left)).long()
            encoder2_attn_mat = torch.tensor(self.pad_sequence(0, encoder2_attn_lst, left=left)).long()
            assert encoder2_input_mat.shape == encoder2_attn_mat.shape
            dct2["encoder2_input_mat"], dct2["encoder2_attn_mat"] = encoder2_input_mat, encoder2_attn_mat

        if len(decoder2_input_id_lst) > 0 and len(decoder2_attn_lst) > 0 and len(decoder2_labels_lst) > 0:
            decoder2_input_mat = torch.tensor(
                self.pad_sequence(self.tokenizer.pad_token_id, decoder2_input_id_lst, left=left)).long()
            decoder2_attn_mat = torch.tensor(self.pad_sequence(0, decoder2_attn_lst, left=left)).long()
            decoder2_labels_mat = torch.tensor(self.pad_sequence(-100, decoder2_labels_lst, left=left)).long()
            assert decoder2_input_mat.shape == decoder2_attn_mat.shape == decoder2_labels_mat.shape
            dct2["decoder2_input_mat"], dct2["decoder2_attn_mat"], \
                dct2["decoder2_labels_mat"] = decoder2_input_mat, decoder2_attn_mat, decoder2_labels_mat
        return {**dct1, **dct2}

    def get_gen_example_gpt2(self, ky_lst):
        input_lst, label_lst, attn_lst = [], [], []
        for i in ky_lst:
            inp = self.data_dict[i]["input_ids_gpt2"] + \
                  self.data_dict[i][self.configuration["experiment_details"]["control"] + "_ids_gpt2"] + \
                  self.data_dict[i]["output_ids_gpt2"]
            lbl = [-100] * (len(self.data_dict[i]["input_ids_gpt2"] +
                                self.data_dict[i][self.configuration["experiment_details"]["control"] + "_ids_gpt2"]) +
                            1) + self.data_dict[i]["output_ids_gpt2"][1:]
            input_lst.append(inp)
            label_lst.append(lbl)
            attn_lst.append([1] * len(inp))
        input_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, input_lst, left=True)).long()
        label_mat = torch.tensor(self.pad_sequence(-100, label_lst, left=True)).long()
        attn_mat = torch.tensor(self.pad_sequence(0, attn_lst, left=True)).long()
        return input_mat, label_mat, attn_mat

    def get_example_fact_span2(self, ky_lst):
        text_ids, fact_ids, label_lst = [], [], []
        for ix in ky_lst:
            text_ids.append(self.data_dict[ix]["model_1_format"]["text_ids"])
            fact_ids.append(self.data_dict[ix]["model_1_format"]["fact_ids"])
            label_lst.append(self.data_dict[ix]["model_1_format"]["labels"])

        text_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, text_ids)).long()
        fact_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, fact_ids)).long()
        label_mat = torch.tensor(self.pad_sequence_3d(-1, label_lst)).long()
        # input_mat = torch.cat([text_mat, fact_mat], -1)
        return text_mat, fact_mat, label_mat

    def make_dataloader(self, ky_lst, distributed=False, batch_size=None):
        if self.configuration["model_type"].startswith("arg_gen_M3"):
            input_dct = self.get_gen_example(ky_lst)
            if input_dct.get("encoder1_input_mat", None) is None and \
                    input_dct.get("decoder1_input_mat", None) is not None:
                print("Encoder inputs not returned. Hence the model must be decoder only like GPT2!")
                data = TensorDataset(input_dct["decoder1_input_mat"], input_dct["decoder1_attn_mat"],
                                     input_dct["decoder1_labels_mat"])
            else:
                print("Encoder inputs returned. Hence the model must be encoder-decoder like BART!")
                if input_dct.get("encoder2_input_mat", None) is None:
                    print("Encoder 2 inputs not returned. Hence the model must only implement 1 encoding-decoding step!")
                    data = TensorDataset(input_dct["encoder1_input_mat"], input_dct["encoder1_attn_mat"],
                                         input_dct["decoder1_input_mat"], input_dct["decoder1_attn_mat"],
                                         input_dct["decoder1_labels_mat"])
                else:
                    data = TensorDataset(input_dct["encoder1_input_mat"], input_dct["encoder1_attn_mat"],
                                         input_dct["decoder1_input_mat"], input_dct["decoder1_attn_mat"],
                                         input_dct["decoder1_labels_mat"],
                                         input_dct["encoder2_input_mat"], input_dct["encoder2_attn_mat"],
                                         input_dct["decoder2_input_mat"], input_dct["decoder2_attn_mat"],
                                         input_dct["decoder2_labels_mat"])

        elif self.configuration["model_type"] == "fact_span_M1":
            text_mat, fact_mat, label_mat = self.get_example_fact_span2(ky_lst)
            assert text_mat.shape[-1] == label_mat.shape[-1]
            data = TensorDataset(text_mat, fact_mat, label_mat)
        elif self.configuration["model_type"] == "span_scheme_M2":
            text_mat, label_scheme, label_bio = self.get_example(ky_lst)
            assert text_mat.shape == label_bio.shape
            data = TensorDataset(text_mat, label_scheme, label_bio)
        else:
            print("INVALID MODEL TYPE!!")
            data = None

        if batch_size is None:
            batch_size = self.configuration["batch_size"]

        if distributed:
            dl = DataLoader(data, batch_size=batch_size, sampler=DistributedSampler(data),
                            num_workers=self.configuration["num_workers"])
        else:
            dl = DataLoader(data, batch_size=batch_size, sampler=SequentialSampler(data),
                            num_workers=self.configuration["num_workers"])
        return dl

    def get_filtered_keys(self, lst, N=-1, filter_keys=None):
        if filter_keys is not None:
            filter_keys = set(filter_keys)
            lst = [i for i in lst if i in filter_keys]
        return lst[:N]

    def get_dataloaders(self, keys_dct, batch_size=None, N=-1, only_test=False, filter_keys=None):
        test_dataloader = self.make_dataloader(self.get_filtered_keys(keys_dct["test"], N=N, filter_keys=filter_keys),
                                               distributed=False, batch_size=batch_size)
        if only_test:
            return test_dataloader
        train_dataloader = self.make_dataloader(self.get_filtered_keys(keys_dct["train"], N=N, filter_keys=filter_keys),
                                                distributed=self.configuration["train_distributed"],
                                                batch_size=batch_size)
        print("Train DataLoader Size:", len(train_dataloader), "| Test Dataloader Size:", len(test_dataloader))
        return train_dataloader, test_dataloader

    def make_generator_test_dataloader_gpt2(self, keys_lst, batch_size=None):
        input_lst, label_lst, attn_lst = [], [], []
        for i in keys_lst:
            inp = self.data_dict[i]["input_ids_gpt2"] + \
                  self.data_dict[i][self.configuration["experiment_details"]["control"] + "_ids_gpt2"] + \
                  self.data_dict[i]["output_ids_gpt2"][:1]
            out = self.data_dict[i]["output_ids_gpt2"][1:]
            input_lst.append(inp)
            label_lst.append(out)
            attn_lst.append([1] * len(inp))
        input_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, input_lst, left=True)).long()
        label_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, label_lst, left=True)).long()
        attn_mat = torch.tensor(self.pad_sequence(0, attn_lst, left=True)).long()
        data = TensorDataset(input_mat, attn_mat, label_mat)
        batch_size = self.configuration["batch_size"] if batch_size is None else batch_size
        dl = DataLoader(data, batch_size=batch_size, sampler=SequentialSampler(data),
                        num_workers=self.configuration["num_workers"])
        return dl

    def make_generator_test_dataloader(self, keys_lst=None, input_text=None, ids=None, batch_size=None):
        input_lst, decoder_input_lst, label_lst = [], [], []
        decoder_input_lst2, label_lst2 = [], []
        if keys_lst is not None:
            for i in keys_lst:
                input_lst.append(self.data_dict[i]["input_ids"])
                ctrl_ids = self.data_dict[i][self.configuration["experiment_details"]["control"] + "_ids"] + \
                           [self.tokenizer.bos_token_id]
                decoder_input_lst.append(ctrl_ids)
                label_lst.append(self.data_dict[i]["output_ids"])
                if "multi_decoder" in self.configuration["model_type"]:
                    decoder_input_lst2.append([self.tokenizer.bos_token_id])
                    label_lst2.append(self.data_dict[i]["output_ids2"])
        else:
            input_lst.append(self.tokenizer(input_text, padding=True, max_length=100).input_ids)
            ctrl_ids = self.tokenizer(ids, add_special_tokens=False).input_ids
            decoder_input_lst.append(ctrl_ids)
            label_lst.append(ctrl_ids)

        input_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, input_lst)).long()
        decoder_input_mat = torch.tensor(decoder_input_lst).long()
        label_mat = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, label_lst)).long()
        if "multi_decoder" in self.configuration["model_type"]:
            decoder_input_mat2 = torch.tensor(decoder_input_lst2).long()
            label_mat2 = torch.tensor(self.pad_sequence(self.tokenizer.pad_token_id, label_lst2)).long()
            data = TensorDataset(input_mat, decoder_input_mat, label_mat, decoder_input_mat2, label_mat2)
        else:
            data = TensorDataset(input_mat, decoder_input_mat, label_mat)

        batch_size = self.configuration["batch_size"] if batch_size is None else batch_size
        dl = DataLoader(data, batch_size=batch_size, sampler=SequentialSampler(data),
                        num_workers=self.configuration["num_workers"])
        return dl

    def get_model(self):
        if self.configuration["model_type"].startswith("arg_gen_M3"):
            if "bart" in self.configuration["experiment_details"]["base_model"]:
                tokenizer = BartTokenizer.from_pretrained(self.configuration["experiment_details"]["base_model"])
                config1 = BartConfig.from_pretrained(self.configuration["experiment_details"]["base_model"])
                if self.configuration["model_type"] == "arg_gen_M3":
                    config2 = BartConfig.from_pretrained(self.configuration["experiment_details"]["base_model"])
                else:
                    config2 = None
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(self.configuration["experiment_details"]["base_model"],
                                                          padding_side="left", truncation_side="left")
                config1 = GPT2Config.from_pretrained(self.configuration["experiment_details"]["base_model"])
                config2 = None
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                config1.pad_token_id = config1.eos_token_id

            config1.output_hidden_states, config1.output_scores, config1.use_cache = True, True, True
            if config2 is not None:
                config2.output_hidden_states, config2.output_scores, config2.use_cache = True, True, True

            self.configuration["in_dim"] = config1.hidden_size
            if self.configuration["add_special_tokens"]:
                num_added_toks = tokenizer.add_special_tokens(self.special_tokens_dict)
                print("Added", num_added_toks, "NEW tokens to tokenizer!")

            self.configuration["tokenizer"] = tokenizer
            if "bart" in self.configuration["experiment_details"]["base_model"]:
                base_model1 = BartForConditionalGeneration.from_pretrained(
                    self.configuration["experiment_details"]["base_model"],
                    config=config1)
                if self.configuration["model_type"] == "arg_gen_M3":
                    base_model2 = BartForConditionalGeneration.from_pretrained(
                        self.configuration["experiment_details"]["base_model"],
                        config=config2)
                else:
                    base_model2 = None
            else:
                base_model1 = GPT2LMHeadModel.from_pretrained(self.configuration["experiment_details"]["base_model"],
                                                              config=config1)
                base_model2 = None
            if self.configuration["add_special_tokens"]:
                base_model1.resize_token_embeddings(len(tokenizer))
                if base_model2 is not None:
                    base_model2.resize_token_embeddings(len(tokenizer))

            if "bart" in self.configuration["experiment_details"]["base_model"]:
                model = EncoderDecoderArgGenerator_M3(base_model1=base_model1, base_model2=base_model2,
                                                      configuration=self.configuration)
            else:
                model = DecoderOnlyArgGenerator_M3(base_model1=base_model1, configuration=self.configuration)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(self.configuration["mname"])
            config = RobertaConfig.from_pretrained(self.configuration["mname"])
            config.output_hidden_states = True
            self.configuration["in_dim"] = config.hidden_size

            if self.configuration["add_special_tokens"] and self.special_tokens_dict is not None:
                num_added_toks = tokenizer.add_special_tokens(self.special_tokens_dict)
            else:
                num_added_toks = 0

            self.configuration["tokenizer"] = tokenizer
            base_model = RobertaModel.from_pretrained(self.configuration["mname"], config=config)

            if self.configuration["add_special_tokens"] and num_added_toks > 0:
                base_model.resize_token_embeddings(len(tokenizer))

            if self.configuration["model_type"] == "fact_span_M1":
                model = FactSpanIdentifier_M1(base_model, self.configuration)
            else:
                if self.configuration["experiment_details"]["pipelined"]:
                    model = SpanSchemePipelinedPredictor_M2(base_model, self.configuration)
                else:
                    model = SpanSchemePredictor_M2(base_model, self.configuration)

        model = model.to(self.device)

        return tokenizer, model

    def reset_model(self):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.configuration["device_num"]}
        if not self.configuration["experiment_details"].get("use_pc", True):
            self.ddp_model.load_state_dict(torch.load(self.configuration["temp_model_name"], map_location=map_location))
        else:
            self.ddp_model.load_state_dict(torch.load(self.configuration["model_name"], map_location=map_location))
        print("DDP Model RE-loaded")
        print(f'The model has {self.count_parameters(self.ddp_model):,} trainable parameters')

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def set_random_seeds(self, random_seed=0):
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def classification_stats_suite(self, tgt_t, src_t, typ, ignore_val=-1, prnt=True):
        if len(tgt_t) == 0 or len(src_t) == 0:
            return [[typ, None, None, None]]
        tgt, src = [], []
        for ix, i in enumerate(tgt_t):
            if i != ignore_val:
                tgt.append(i)
                src.append(src_t[ix])
        assert len(tgt) == len(src)
        cm = confusion_matrix(tgt, src)
        cs = classification_report(tgt, src, zero_division=0)
        cs_dict = classification_report(tgt, src, zero_division=0, output_dict=True)
        if prnt:
            print("\n===== STATS FOR ", typ, "=====")
            print("Confusion metric : \n", cm)
            print("Classification Stats:\n", cs)
            print("==============================\n")

        stat_lst = []  # Task, Label, Metric, Score
        for k, v in cs_dict.items():
            if k == 'accuracy':
                stat_lst.append([typ, "overall", k, v])
            else:
                stat_lst.append([typ, k, "f1-score", v["f1-score"]])
                stat_lst.append([typ, k, "precision", v["precision"]])
                stat_lst.append([typ, k, "recall", v["recall"]])
                stat_lst.append([typ, k, "support", v["support"]])
        return stat_lst

    def make_model_input_dict(self, batch):
        if self.configuration["model_type"].startswith("arg_gen_M3"):
            return_dict = {}
            if "gpt2" in self.configuration["experiment_details"]["base_model"]:
                return_dict["decoder1_input_mat"], return_dict["decoder1_attn_mat"], \
                    return_dict["decoder1_labels_mat"] = batch[0], batch[1], batch[2]
            else:
                return_dict["encoder1_input_mat"], return_dict["encoder1_attn_mat"], \
                return_dict["decoder1_input_mat"], return_dict["decoder1_attn_mat"], \
                return_dict["decoder1_labels_mat"] = batch[0], batch[1], batch[2], batch[3], batch[4]
                if len(batch) > 5:
                    return_dict["encoder2_input_mat"], return_dict["encoder2_attn_mat"], \
                    return_dict["decoder2_input_mat"], return_dict["decoder2_attn_mat"], \
                    return_dict["decoder2_labels_mat"] = batch[5], batch[6], batch[7], batch[8], batch[9]
            return return_dict
        elif self.configuration["model_type"] == "fact_span_M1":
            text_mat, fact_mat, label_mat = batch
            return {"text_ids": text_mat, "fact_ids": fact_mat, "labels": label_mat}
        elif self.configuration["model_type"] == "span_scheme_M2":
            text_mat, label_scheme, label_bio = batch
            return {"input_ids": text_mat, "span_labels": label_bio, "scheme_labels": label_scheme}
        else:
            return {}

    def train(self, dataloader):
        self.ddp_model.train()
        ep_t_loss, batch_num = 0, 0
        ep_t_loss_m1, ep_t_loss_m2 = 0.0, 0.0

        self.optimizer.zero_grad()
        for ix, batch in tqdm(enumerate(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_dict = self.make_model_input_dict(batch)
            if self.configuration["model_type"] == "span_scheme_M2" and \
                    self.configuration["experiment_details"]["pipelined"]:
                input_dict["use_pred_attn_mask"] = False

            with autocast():
                output_dct = self.ddp_model(input_dict=input_dict)
                loss = output_dct["loss"]
                batch_num += 1
                loss = loss / self.configuration["accumulate"]

                ep_t_loss += loss.item()
                ep_t_loss_m1 += output_dct["model1_loss"].item()
                if output_dct.get("model2_loss", None) is not None:
                    ep_t_loss_m2 += output_dct["model2_loss"].item()

            self.scaler.scale(loss).backward()

            if (ix + 1) % self.configuration["accumulate"] == 0 or ix + 1 == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        return ep_t_loss / batch_num, ep_t_loss_m1 / batch_num, ep_t_loss_m2 / batch_num

    def class_wise_stats(self, tgt, pred):
        pred, tgt = np.asarray(pred), np.asarray(tgt)
        stat_list = []
        for idx in range(pred.shape[1]):
            s = self.classification_stats_suite(tgt[:, idx], pred[:, idx],
                                                "Scheme Prediction Class: " + mappings.id_to_scheme_new[idx].upper(),
                                                ignore_val=-1, prnt=False)
            stat_list.extend(s)
        return stat_list

    def resolve_span_label(self, lst):
        lst = list(set([mappings.id_to_span_type[i].split("-")[-1] for i in lst]))
        assert len(lst) == 1
        return lst[0]

    def get_contigious_span_from_list(self, lst):
        op, tmp, tmp_lbl, span_lbl = [], [], [], []
        for ix, i in enumerate(lst):
            if i != 0:
                if i in [1, 3, 5, 7, 9, 11]:
                    if len(tmp) > 0:
                        if i == lst[ix - 1]:
                            tmp.append(ix)
                            tmp_lbl.append(i)
                        else:
                            op.append(tmp)
                            span_lbl.append(self.resolve_span_label(tmp_lbl))
                            tmp, tmp_lbl = [ix], [i]
                    else:
                        tmp, tmp_lbl = [ix], [i]
                else:
                    if len(tmp) > 0:
                        if i == lst[ix - 1] or i - lst[ix - 1] == 1:
                            tmp.append(ix)
                            tmp_lbl.append(i)
                        else:
                            op.append(tmp)
                            span_lbl.append(self.resolve_span_label(tmp_lbl))
                            tmp, tmp_lbl = [ix], [i]
                    else:
                        tmp.append(ix)
                        tmp_lbl.append(i)
            else:
                if len(tmp) > 0:
                    op.append(tmp)
                    span_lbl.append(self.resolve_span_label(tmp_lbl))
                    tmp, tmp_lbl = [], []

        if len(tmp) > 0:
            op.append(tmp)
            span_lbl.append(self.resolve_span_label(tmp_lbl))

        assert len(span_lbl) == len(op)

        return op, span_lbl

    def get_lcs(self, lst_a, lst_b):
        s = SequenceMatcher(None, lst_a, lst_b)
        try:
            lcs_a, _, lcs_size = s.find_longest_match(0, len(lst_a), 0, len(lst_b))
            return lst_a[lcs_a: lcs_a + lcs_size]
        except Exception as e:
            print("Exception", e)
            return []

    def get_text(self, text_lst, lst):
        return self.tokenizer.decode([text_lst[ix] for ix in lst], skip_special_tokens=True).strip()

    def evaluate_span(self, text_lst, pred_lst, tgt_lst=[]):
        """
        text_lst: text tokens
        pred_lst: list of predicted tag lists
        tgt_lst: list of target tag lists
        """
        overlap_result, pred_span_texts_all = [], []
        for pred_ix, pred in enumerate(pred_lst):
            if len(tgt_lst) > 0:
                pred = [i for ix, i in enumerate(pred) if tgt_lst[pred_ix][ix] != -1]
                tgt = [i for i in tgt_lst[pred_ix] if i != -1]
                tgt_contig_spans, tgt_span_lbls = self.get_contigious_span_from_list(tgt)
            else:
                tgt_contig_spans = None

            pred_contig_spans, pred_span_lbls = self.get_contigious_span_from_list(pred)
            pred_span_texts = [self.get_text(text_lst[pred_ix], i) for i in pred_contig_spans]
            orig_text = self.tokenizer.decode(text_lst[pred_ix], skip_special_tokens=True).strip()
            pred_span_texts_all.append([pred_ix, orig_text, pred_span_texts])

            if tgt_contig_spans is not None:
                used_preds = []
                for ts_idx, ts in enumerate(tgt_contig_spans):
                    mx_ps, mx_p_idx = 0, None
                    for p_idx, ps in enumerate(pred_contig_spans):
                        lcs = self.get_lcs(ts, ps)
                        if len(lcs) >= mx_ps:
                            mx_ps = len(lcs)
                            mx_p_idx = p_idx
                    """ tgt span, best matching pred span, tgt txt, pred txt, len of pred span, 
                        overlap wrt tgt, overlap wrt self """
                    ts_txt = self.get_text(text_lst[pred_ix], ts)
                    ts_lbl = tgt_span_lbls[ts_idx]
                    if mx_p_idx is None:
                        overlap_result.append([pred_ix, ts, None, ts_txt, None, ts_lbl, None, 0, 0, 0])
                    else:
                        p_txt = pred_span_texts[mx_p_idx]  # self.get_text(text_lst, pred_contig_spans[mx_p_idx])
                        p_lbl = pred_span_lbls[mx_p_idx]
                        used_preds.append(mx_p_idx)
                        overlap_result.append([pred_ix, ts, pred_contig_spans[mx_p_idx], ts_txt, p_txt, ts_lbl, p_lbl,
                                               mx_ps, mx_ps / len(ts), mx_ps / len(pred_contig_spans[mx_p_idx])])
                for ix, i in enumerate(pred_contig_spans):
                    if ix not in used_preds:
                        overlap_result.append([pred_ix, None, i, None, pred_span_texts[ix], None, pred_span_lbls[ix],
                                               len(i), 0.0, 1.0])

        tmpdf_span = pd.DataFrame(overlap_result, columns=["id", "tgt_span", "matched_pred_span", "tgt_txt", "pred_txt",
                                                           "tgt_lbl", "pred_lbl", "len_pred_span", "tgt_overlap",
                                                           "self_overlap"])
        # print(tmpdf_span)
        partial_matches = (tmpdf_span["tgt_overlap"] >= 0.5) & (tmpdf_span["self_overlap"] >= 0.5)
        full_matches = (tmpdf_span["tgt_overlap"] == 1) & (tmpdf_span["self_overlap"] == 1)

        t_lbl_part, t_lbl_full = set(tmpdf_span[partial_matches]["tgt_lbl"]), set(tmpdf_span[full_matches]["tgt_lbl"])
        if len(t_lbl_part) == 1:
            span_stats_part = None
            print("Not calculating Partial classification stats as target has only 1 Label:", t_lbl_part)
        else:
            span_stats_part = self.classification_stats_suite(list(tmpdf_span[partial_matches]["tgt_lbl"]),
                                                              list(tmpdf_span[partial_matches]["pred_lbl"]),
                                                              "Span Label Partial", ignore_val=-1)
        if len(t_lbl_full) == 1:
            span_stats_full = None
            print("Not calculating Full classification stats as target has only 1 Label:", t_lbl_full)
        else:
            span_stats_full = self.classification_stats_suite(list(tmpdf_span[full_matches]["tgt_lbl"]),
                                                              list(tmpdf_span[full_matches]["pred_lbl"]),
                                                              "Span Label Full", ignore_val=-1)

        tp_partial, tp_full = partial_matches.sum(), full_matches.sum()

        fn_partial = ((tmpdf_span["tgt_overlap"] < 0.5) | (tmpdf_span["self_overlap"] < 0.5)).sum()
        fn_full = ((tmpdf_span["tgt_overlap"] != 1) & (tmpdf_span["self_overlap"] != 1)).sum()

        fp = tmpdf_span.tgt_span.isna().sum()

        f1_partial = tp_partial / (tp_partial + 0.5 * (fn_partial + fp))
        f1_full = tp_full / (tp_full + 0.5 * (fn_full + fp))
        res = [["Span Identification Partial", "Span", "f1-score", f1_partial],
               ["Span Identification Full", "Span", "f1-score", f1_full]]
        if span_stats_part is not None:
            res.extend(span_stats_part)
        if span_stats_full is not None:
            res.extend(span_stats_full)

        return {"pred_span_texts": pred_span_texts_all, "overlap_result": res, "calculation_df": tmpdf_span}

    def evaluate(self, dataloader):
        ep_t_loss, ep_t_loss_m1, ep_t_loss_m2, batch_num = 0, 0.0, 0.0, 0
        scheme_pred_binary, scheme_tgt_binary = [], []
        scheme_pred_class, scheme_tgt_class = [], []
        span_pred, span_tgt = [], []
        txt_lst, span_pred_lst, span_tgt_lst = [], [], []
        stat_dict = {}

        for ix, batch in tqdm(enumerate(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_dict = self.make_model_input_dict(batch)
            if self.configuration["model_type"] == "span_scheme_M2" and \
                    self.configuration["experiment_details"]["pipelined"]:
                input_dict["use_pred_attn_mask"] = True

            with autocast():
                with torch.no_grad():
                    self.model.eval()
                    output_dct = self.model(input_dict=input_dict)
                    loss = output_dct["loss"]
                    ep_t_loss += loss.item()
                    ep_t_loss_m1 += output_dct["model1_loss"].item()
                    if output_dct.get("model2_loss", None) is not None:
                        ep_t_loss_m2 += output_dct["model2_loss"].item()
                    batch_num += 1

            if output_dct.get("binary_logits", None) is not None or output_dct.get("multiclass_logits",
                                                                                   None) is not None:
                s_pred = (torch.sigmoid(output_dct["binary_logits"]) >=
                          self.configuration["experiment_details"]["threshold"]).long().detach().cpu() \
                    if output_dct.get("binary_logits", None) is not None \
                    else torch.argmax(output_dct["multiclass_logits"], -1).detach().cpu()
                span_pred.extend(s_pred.view(-1).tolist())
                span_tgt.extend(input_dict["labels"].view(-1).tolist())

            if output_dct.get("scheme_logits", None) is not None:
                s_pred = (torch.sigmoid(output_dct["scheme_logits"]) >=
                          self.configuration["experiment_details"]["threshold"]).long().detach().cpu()
                scheme_pred_binary.extend(s_pred.view(-1).tolist())
                ky = "scheme_labels" if self.configuration["model_type"] == "span_scheme_M2" else "labels"
                scheme_tgt_binary.extend(input_dict[ky].view(-1).tolist())

                scheme_pred_class.extend(s_pred.tolist())
                scheme_tgt_class.extend(input_dict[ky].tolist())

            if output_dct.get("span_logits", None) is not None:
                s_pred = torch.argmax(output_dct["span_logits"], -1).detach().cpu()
                span_pred.extend(s_pred.view(-1).tolist())
                ky = "span_labels" if self.configuration["model_type"] == "span_scheme_M2" else "labels"
                span_tgt.extend(input_dict[ky].view(-1).tolist())

                span_pred_lst.extend(s_pred.tolist())
                span_tgt_lst.extend(input_dict[ky].tolist())
                txt_lst.extend(input_dict["input_ids"].tolist())

        s_stats_bin, s_stats_class, a_stats, a_stats_class = None, None, None, None
        span_stats, span_eval = None, None
        if "scheme" in self.configuration["model_type"]:
            s_stats_bin = self.classification_stats_suite(scheme_tgt_binary, scheme_pred_binary,
                                                          "Scheme Prediction Overall Binary", ignore_val=-1)
            s_stats_class = self.class_wise_stats(scheme_tgt_class, scheme_pred_class)

        if "span" in self.configuration["model_type"]:
            span_stats = self.classification_stats_suite(span_tgt, span_pred, "Span Prediction", ignore_val=-1)
            if len(span_pred_lst) > 0:
                span_eval = self.evaluate_span(txt_lst, span_pred_lst, span_tgt_lst)

        stat_dict["scheme_stats_binary"] = s_stats_bin
        stat_dict["scheme_stats_by_class"] = s_stats_class
        stat_dict["span_stats"] = span_stats
        stat_dict["span_eval"] = span_eval

        return ep_t_loss / batch_num, ep_t_loss_m1 / batch_num, ep_t_loss_m2 / batch_num, stat_dict

    def compute_generation_metrics(self, hyp, cor, rouge, metadata=None):
        if len(cor) <= 0:
            return {"hyp": hyp if len(hyp) > 0 else None}

        assert len(hyp) == len(cor)
        references = [[i.strip().split()] for i in cor]
        candidates = [i.strip().split() for i in hyp]
        bleu_score = corpus_bleu(references, candidates)
        bleu_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, candidates, weights=(0, 1, 0, 0))
        bleu_3 = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
        bleu_4 = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))

        rouge_score = rouge.compute(predictions=hyp, references=cor)
        rouge1, rouge2, rougeL, rougeLsum = rouge_score["rouge1"].mid.fmeasure, rouge_score["rouge2"].mid.fmeasure, \
                                            rouge_score["rougeL"].mid.fmeasure, rouge_score["rougeLsum"].mid.fmeasure
        metadata = "summary_stats" if metadata is None else metadata
        test_scores = {"bleu": bleu_score, "bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
                       "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rougeLsum": rougeLsum,
                       "summary": [metadata, bleu_score, bleu_4, rougeL], "cor": cor, "hyp": hyp}
        return test_scores

    def remove_special_tokens(self, text, level="all"):
        special_keys_wo_var = ["<"+"_".join(i.split())+">" for i in mappings.generator_token_M3.keys() if "VAR" not in i]
        model_specific_keys = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]
        ky_list = special_keys_wo_var + model_specific_keys if level == "all" else special_keys_wo_var if level == "special" else model_specific_keys
        for k in ky_list:
            text = text.replace(k, "")
        return " ".join(text.strip().split())

    def get_pattern_and_argument(self, txt_lst, original=False):
        pattern_list, argument_list, original_list = [], [], []

        for txt in txt_lst:
            txt = " ".join(txt.replace(self.tokenizer.pad_token, "").split())
            original_list.append(txt)
            if original:
                pattern, argument = txt.split("<argument>")
                pattern = pattern.split("<pattern>")[1]
            else:
                if "<argument>" in txt:
                    pattern, argument = txt.split("<argument>", 1)
                    pattern = pattern.split("<pattern>")[1]
                else:
                    pattern = argument = txt.split("<pattern>")[1]
            pattern_list.append(self.remove_special_tokens(pattern.strip()))
            argument_list.append(self.remove_special_tokens(argument.strip()))
        assert len(pattern_list) == len(argument_list) == len(original_list)
        return pattern_list, argument_list, original_list

    def get_pattern(self, txt_lst):
        pattern_list, argument_list, original_list = [], [], []

        for txt in txt_lst:
            txt = " ".join(txt.replace(self.tokenizer.pad_token, "").split())
            original_list.append(txt)
            if "<pattern>" in txt and "<argument>" in txt:
                pattern, argument = txt.split("<argument>", 1)
                pattern = pattern.split("<pattern>")[1]
            elif "<pattern>" in txt or "<argument>" in txt:
                pattern = argument = re.split(r"<pattern>|<argument>", txt, maxsplit=1)[-1]
            else:
                pattern = argument = ""
            pattern_list.append(self.remove_special_tokens(pattern.strip()))
            argument_list.append(self.remove_special_tokens(argument.strip()))
        assert len(pattern_list) == len(argument_list) == len(original_list)
        return pattern_list, argument_list, original_list

    def generate_response(self, dataloader, rouge, config_dict={}):
        self.model.eval()
        cor_full, hyp_full, cor_pat, hyp_pat, cor_arg, hyp_arg = [], [], [], [], [], []

        for ix, batch in tqdm(enumerate(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_dict = self.make_model_input_dict(batch)
            if input_dict.get("encoder2_input_mat", None) is None:
                orig = input_dict["decoder1_input_mat"]
            else:
                orig = torch.cat([input_dict["encoder2_input_mat"], input_dict["decoder2_input_mat"]], 1)
            orig = self.tokenizer.batch_decode([[j for j in i.tolist() if j != self.tokenizer.pad_token_id]
                                                for i in orig])
            with torch.no_grad():
                res = self.model.predict(input_dict=input_dict, config_dict=config_dict)
                # res_pat_list, res_arg_list, res_pred_list = self.get_pattern_and_argument(res, original=False)
                # orig_pat_list, orig_arg_list, orig_pred_list = self.get_pattern_and_argument(orig, original=True)
                res_pat_list, res_arg_list, res_pred_list = self.get_pattern(res)
                orig_pat_list, orig_arg_list, orig_pred_list = self.get_pattern(orig)

                cor_full.extend(orig_pred_list)
                cor_pat.extend(orig_pat_list)
                cor_arg.extend(orig_arg_list)
                hyp_full.extend(res_pred_list)
                hyp_pat.extend(res_pat_list)
                hyp_arg.extend(res_arg_list)

        test_scores_full = self.compute_generation_metrics(hyp_full, cor_full, rouge, "full_decoding")
        test_scores_pat = self.compute_generation_metrics(hyp_pat, cor_pat, rouge, "pattern_decoding")
        test_scores_arg = self.compute_generation_metrics(hyp_arg, cor_arg, rouge, "argument_decoding")
        return {"full_decoding": test_scores_full, "test_scores_pat": test_scores_pat,
                "test_scores_arg": test_scores_arg}

    def generate_single(self, enc_input_ids, enc_attn_mask, dec_input_ids, dec_attn_mask,
                        config_dict={}, target="pattern"):

        var_ids = [self.tokenizer.get_vocab()["<" + i + ">"] for i in mappings.variable_list]
        vars_absent = [[i] for i in var_ids if i not in enc_input_ids]

        if target == "pattern":
            if len(vars_absent) > 0:
                res = self.model.base_model1.generate(input_ids=enc_input_ids,  attention_mask=enc_attn_mask,
                                                      decoder_input_ids=dec_input_ids,
                                                      decoder_attention_mask=dec_attn_mask,
                                                      do_sample=config_dict.get("do_sample", False),
                                                      early_stopping=config_dict.get("early_stopping", True),
                                                      no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                                      num_beams=config_dict.get("num_beams", 5),
                                                      bad_words_ids=vars_absent, return_dict_in_generate=True,
                                                      num_return_sequences=config_dict.get("num_return_sequences", 1),
                                                      max_new_tokens=config_dict.get("max_length", 50))
            else:
                res = self.model.base_model1.generate(input_ids=enc_input_ids, attention_mask=enc_attn_mask,
                                                      decoder_input_ids=dec_input_ids,
                                                      decoder_attention_mask=dec_attn_mask,
                                                      do_sample=config_dict.get("do_sample", False),
                                                      early_stopping=config_dict.get("early_stopping", True),
                                                      no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                                      num_beams=config_dict.get("num_beams", 5),
                                                      return_dict_in_generate=True,
                                                      num_return_sequences=config_dict.get("num_return_sequences", 1),
                                                      max_new_tokens=config_dict.get("max_length", 50))
        else:
            res = self.model.base_model1.generate(input_ids=enc_input_ids, attention_mask=enc_attn_mask,
                                                  decoder_input_ids=dec_input_ids,
                                                  decoder_attention_mask=dec_attn_mask,
                                                  do_sample=config_dict.get("do_sample", False),
                                                  early_stopping=config_dict.get("early_stopping", True),
                                                  no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                                  num_beams=config_dict.get("num_beams", 5),
                                                  bad_words_ids=[[i] for i in var_ids], return_dict_in_generate=True,
                                                  num_return_sequences=config_dict.get("num_return_sequences", 1),
                                                  max_new_tokens=config_dict.get("max_length", 50))

        responses = [[j for j in i.tolist() if j != self.tokenizer.pad_token_id] for ix, i in enumerate(res["sequences"])]
        return responses #self.tokenizer.batch_decode(responses)[0]

    def generate_response_V2(self, keys_dict, rouge, config_dict={}):
        n_prompt_tokens = 2 + len(self.configuration["experiment_details"]["control"].split("_"))
        eos, pad, bos, argu = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,\
                                self.tokenizer.bos_token_id, self.tokenizer.get_vocab()["<argument>"]
        res_op_dict = {}
        cor_pat, hyp_pat, cor_arg, hyp_arg = [], [], [], []
        for k in tqdm(keys_dict):
            """ 
            encoder1_input_ids: <s> Topic <VAR_x> Text ... </s>
            decoder1_input_ids: <s> <CTRL_x> <CTRL_y> <argument/pattern>
            encoder2_input_ids: <s> Topic <VAR_x> Text ... <CTRL_x> <CTRL_y> <pattern> Pattern Text </s>
            """
            d = {}
            exp_id = self.configuration["experiment_details"].get("alt_id",
                                                                  self.configuration["experiment_details"]["exp_id"])
            d_n = self.data_dict[k]["model_3_format"]["bart"]["V2_" + str(exp_id)]
            if not self.configuration["experiment_details"]["multi_encoder"]:
                # d0 = self.data_dict[k]["model_3_format"]["bart"]["V2_0"]
                encoder1_input_ids = torch.tensor(d_n["arg_encoder_input_ids"]).unsqueeze(0).to(self.device)
                encoder1_attn_mask = torch.ones_like(encoder1_input_ids)

                decoder1_input_ids = torch.tensor(d_n["arg_decoder_input_ids"][:n_prompt_tokens]).unsqueeze(0).to(
                    self.device)
                decoder1_attn_mask = torch.ones_like(decoder1_input_ids)

                if self.configuration["experiment_details"]["exp_id"] == 1:
                    pat_end_ids = [ix for ix, i in enumerate(d_n["arg_decoder_labels"]) if i == eos][0]
                    pat_orig_labels = d_n["arg_decoder_labels"][n_prompt_tokens - 1: pat_end_ids]
                    pat_gen = self.generate_single(encoder1_input_ids, encoder1_attn_mask, decoder1_input_ids,
                                                   decoder1_attn_mask, config_dict=config_dict, target="pattern")
                    pat_gen_labels = [i for i in pat_gen[0][n_prompt_tokens:] if i not in [eos, pad]]

                    addn = [eos, bos, argu] if pat_gen[0][-1] != eos else [bos, argu]
                    decoder2_input_ids = torch.tensor(pat_gen[0] + addn).unsqueeze(0).to(self.device)
                    decoder2_attn_mask = torch.ones_like(decoder2_input_ids)
                    arg_orig_labels = d_n["arg_decoder_labels"][pat_end_ids+3: -1]

                    arg_gen = self.generate_single(encoder1_input_ids, encoder1_attn_mask, decoder2_input_ids,
                                                   decoder2_attn_mask, config_dict=config_dict, target="argument")
                    arg_gen_labels = [i for i in arg_gen[0][decoder2_input_ids.shape[-1]:] if i not in [eos, pad]]

                    d["pat_encoder_input_ids"], d["pat_decoder_input_ids"] = None, None
                    d["pat_orig_labels"], d["pat_gen_labels"] = pat_orig_labels, pat_gen_labels
                    d["pat_orig_text"], d["pat_gen_text"] = self.tokenizer.decode(d["pat_orig_labels"]), \
                                                            self.tokenizer.decode(d["pat_gen_labels"])
                    d["arg_encoder_input_ids"], d["arg_decoder_input_ids"] = encoder1_input_ids, decoder1_input_ids
                    d["arg_orig_labels"], d["arg_gen_labels"] = arg_orig_labels, arg_gen_labels
                    d["arg_orig_text"], d["arg_gen_text"] = self.tokenizer.decode(d["arg_orig_labels"]), \
                                                            self.tokenizer.decode(d["arg_gen_labels"])
                    cor_arg.append(d["arg_orig_text"])
                    hyp_arg.append(d["arg_gen_text"])
                    cor_pat.append(d["pat_orig_text"])
                    hyp_pat.append(d["pat_gen_text"])
                else:
                    arg_orig_labels = d_n["arg_decoder_labels"][n_prompt_tokens - 1: -1]
                    arg_gen = self.generate_single(encoder1_input_ids, encoder1_attn_mask, decoder1_input_ids,
                                                   decoder1_attn_mask, config_dict=config_dict, target="argument")
                    arg_gen_labels = arg_gen[0][n_prompt_tokens:]

                    d["arg_encoder_input_ids"], d["arg_decoder_input_ids"] = encoder1_input_ids, decoder1_input_ids
                    d["arg_orig_labels"], d["arg_gen_labels"] = arg_orig_labels, arg_gen_labels
                    d["arg_orig_text"], d["arg_gen_text"] = self.tokenizer.decode(d["arg_orig_labels"]), \
                                                            self.tokenizer.decode(d["arg_gen_labels"])
                    cor_arg.append(d["arg_orig_text"])
                    hyp_arg.append(d["arg_gen_text"])
                    d["pat_encoder_input_ids"], d["pat_decoder_input_ids"], d["pat_orig_labels"], \
                        d["pat_gen_labels"], d["pat_orig_text"], d["pat_gen_text"] = None, None, None, None, None, None

            else:
                # d1 = self.data_dict[k]["model_3_format"]["bart"]["V2_1"]
                encoder1_input_ids = torch.tensor(d_n["pat_encoder_input_ids"]).unsqueeze(0).to(self.device)
                encoder1_attn_mask = torch.ones_like(encoder1_input_ids)

                decoder1_input_ids = torch.tensor(d_n["pat_decoder_input_ids"][:n_prompt_tokens]).unsqueeze(0).to(
                    self.device)
                decoder1_attn_mask = torch.ones_like(decoder1_input_ids)

                pat_orig_labels = d_n["pat_decoder_labels"][n_prompt_tokens - 1: -1]
                pat_gen = self.generate_single(encoder1_input_ids, encoder1_attn_mask, decoder1_input_ids,
                                               decoder1_attn_mask, config_dict=config_dict, target="pattern")
                pat_gen_labels = pat_gen[0][n_prompt_tokens:]

                encoder2_input_ids = torch.cat([encoder1_input_ids[:, :-1],
                                                torch.tensor(pat_gen)[:, 1:].to(self.device)], -1)#, encoder1_input_ids[:, -1:]], -1)
                encoder2_attn_mask = torch.ones_like(encoder2_input_ids)

                decoder2_input_ids = torch.tensor(d_n["arg_decoder_input_ids"][:n_prompt_tokens]).unsqueeze(0).to(
                    self.device)
                decoder2_attn_mask = torch.ones_like(decoder2_input_ids)
                arg_orig_labels = d_n["arg_decoder_labels"][n_prompt_tokens - 1: -1]
                arg_gen = self.generate_single(encoder2_input_ids, encoder2_attn_mask, decoder2_input_ids,
                                               decoder2_attn_mask, config_dict=config_dict, target="argument")
                arg_gen_labels = arg_gen[0][n_prompt_tokens:]

                d["pat_encoder_input_ids"], d["pat_decoder_input_ids"] = encoder1_input_ids, decoder1_input_ids
                d["pat_orig_labels"], d["pat_gen_labels"] = pat_orig_labels, pat_gen_labels
                d["pat_orig_text"], d["pat_gen_text"] = self.tokenizer.decode(d["pat_orig_labels"]), \
                                                        self.tokenizer.decode(d["pat_gen_labels"])
                d["arg_encoder_input_ids"], d["arg_decoder_input_ids"] = encoder2_input_ids, decoder2_input_ids
                d["arg_orig_labels"], d["arg_gen_labels"] = arg_orig_labels, arg_gen_labels
                d["arg_orig_text"], d["arg_gen_text"] = self.tokenizer.decode(d["arg_orig_labels"]), \
                                                        self.tokenizer.decode(d["arg_gen_labels"])
                cor_arg.append(d["arg_orig_text"])
                hyp_arg.append(d["arg_gen_text"])
                cor_pat.append(d["pat_orig_text"])
                hyp_pat.append(d["pat_gen_text"])
            res_op_dict[k] = d

        test_scores_pat = self.compute_generation_metrics(hyp_pat, cor_pat, rouge, "pattern_decoding")
        test_scores_arg = self.compute_generation_metrics(hyp_arg, cor_arg, rouge, "argument_decoding")
        return {"full_decoding": None, "test_scores_pat": test_scores_pat,
                "test_scores_arg": test_scores_arg, "response_dict": res_op_dict}
