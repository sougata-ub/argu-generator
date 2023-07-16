import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchcrf import CRF
import utils
import mappings
# import copy
import random


class Encoder(nn.Module):
    def __init__(self, base_encoder, N=4):
        super().__init__()
        self.base_encoder = base_encoder
        self.N = N

    def forward(self, input_ids, attention_mask):
        enc = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # print("enc:", enc["hidden_states"][0].shape, len(enc["hidden_states"]))
        return torch.stack(enc["hidden_states"][-self.N:]).mean(0)  # batch, seq len, h_dim


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h):
        return self.dropout(F.relu(self.ff(h)))


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int,
                 bias: bool = True):  # ,add_attn: bool = False):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
        #         self.add_attn = add_attn
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:  # , attn: torch.Tensor
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):  # , add_attn: bool):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)  # , add_attn=add_attn)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:  # , attn: torch.Tensor
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class FactSpanIdentifier_M1(nn.Module):
    def __init__(self, base_encoder, configuration):
        super().__init__()
        self.configuration = configuration
        self.encoder = Encoder(base_encoder)

        self.mlp_arg = MLP(configuration["in_dim"], configuration["experiment_details"]["out_dim"])
        self.mlp_fact = MLP(configuration["in_dim"], configuration["experiment_details"]["out_dim"])
        self.biaff = Biaffine(configuration["experiment_details"]["out_dim"],
                              configuration["experiment_details"]["out_dim"],
                              configuration["experiment_details"]["n_classes"])

    def forward(self, input_dict):
        text_ids = input_dict["text_ids"]
        fact_ids = input_dict["fact_ids"]
        input_ids = torch.cat([text_ids, fact_ids], -1)
        labels = input_dict.get("labels", None)

        arg_len = text_ids.shape[-1]
        fact_bos_idx = fact_ids == self.configuration["tokenizer"].bos_token_id

        attention_mask = (input_ids != self.configuration["tokenizer"].pad_token_id).long().to(input_ids.device)
        arg_fact_enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # batch, seq, h_dim

        arg_repr = arg_fact_enc[:, :arg_len, :]  # batch, arg_len, h_dim
        fact_repr = arg_fact_enc[:, arg_len:, :]
        batch, _, h_dim = fact_repr.shape
        fact_repr = fact_repr[fact_bos_idx].view(batch, -1, h_dim)  # batch, N_facts, h_dim

        arg_repr_reduced = self.mlp_arg(arg_repr)  # batch, arg_max_len, out_dim
        fact_repr_reduced = self.mlp_fact(fact_repr)  # batch, N_facts, out_dim
        biaff_logits = self.biaff(fact_repr_reduced, arg_repr_reduced)  # (batch, N_facts, arg_max_len, n_classes)

        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-1)
            loss = ce(biaff_logits.contiguous().view(-1, self.configuration["experiment_details"]["n_classes"]),
                      labels.contiguous().view(-1))

        return {"multiclass_logits": biaff_logits, "loss": loss}


class SpanSchemePredictor_M2(nn.Module):
    """ Predicts knowledge span AND arg scheme from text """

    def __init__(self, base_encoder, configuration):
        super().__init__()
        self.configuration = configuration
        self.encoder = Encoder(base_encoder)

        self.span_prediction_head = nn.Linear(self.configuration["in_dim"],
                                              self.configuration["experiment_details"]["n_classes_span"])
        self.scheme_prediction_head = nn.Linear(self.configuration["in_dim"],
                                                self.configuration["experiment_details"]["n_classes_scheme"])

    def forward(self, input_dict, get_representation=False):
        input_ids = input_dict["input_ids"]
        span_labels = input_dict.get("span_labels", None)
        scheme_labels = input_dict.get("scheme_labels", None)

        attention_mask = (input_ids != self.configuration["tokenizer"].pad_token_id).long().to(input_ids.device)
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # batch, seq, h_dim
        enc_pooled = enc.mean(1)
        span_pred = self.span_prediction_head(enc)  # batch, seq, n_classes
        scheme_pred = self.scheme_prediction_head(enc_pooled)  # batch, n_scheme_classes

        loss, scheme_loss, additional_pred = None, None, None
        if span_labels is not None:
            ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            loss = ce_loss(span_pred.view(-1, self.configuration["experiment_details"]["n_classes_span"]),
                           span_labels.view(-1).contiguous().long())

        if scheme_labels is not None:
            bce_loss = nn.BCEWithLogitsLoss()
            scheme_loss = bce_loss(scheme_pred, scheme_labels)
            loss += scheme_loss

        op = {"span_logits": span_pred, "scheme_logits": scheme_pred, "loss": loss}
        if get_representation:
            op["representations"] = enc_pooled
        return op


class SpanSchemePipelinedPredictor_M2(nn.Module):
    def __init__(self, base_encoder, configuration):
        super().__init__()
        self.configuration = configuration
        self.encoder = Encoder(base_encoder)

        self.self_mha = nn.ModuleList([nn.MultiheadAttention(self.configuration["in_dim"],
                                                             self.configuration["experiment_details"]["n_heads"],
                                                             batch_first=True, dropout=0.1)
                                       for _ in range(self.configuration["experiment_details"]["n_self_attn_layers"])])
        self.self_mha_drop = nn.Dropout(0.1)

        self.span_prediction_head = nn.Linear(self.configuration["in_dim"],
                                              self.configuration["experiment_details"]["n_classes_span"])
        self.scheme_prediction_head = nn.Linear(self.configuration["in_dim"],
                                                self.configuration["experiment_details"]["n_classes_scheme"])

    def forward(self, input_dict, get_representation=False):
        input_ids = input_dict["input_ids"]
        span_labels = input_dict.get("span_labels", None)
        scheme_labels = input_dict.get("scheme_labels", None)
        use_pred_attn_mask = input_dict.get("use_pred_attn_mask", True)

        span_attention_mask = (input_ids != self.configuration["tokenizer"].pad_token_id).long().to(input_ids.device)
        enc = self.encoder(input_ids=input_ids, attention_mask=span_attention_mask)  # batch, seq, h_dim
        span_pred = self.span_prediction_head(enc)  # batch, seq, n_classes

        loss, scheme_loss, additional_pred = None, None, None
        if span_labels is not None:
            ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            loss = ce_loss(span_pred.view(-1, self.configuration["experiment_details"]["n_classes_span"]),
                           span_labels.view(-1).contiguous().long())

        '''
        key_padding_mask – (N,S) indicating which elements within key to treat as “padding”. 
        True value indicates that the corresponding key value will be ignored for the purpose of attention. 
        For a byte mask, a non-zero value indicates that the corresponding key value will be ignored.'''
        pred_key_padding_mask = ((span_pred.argmax(-1) == 0).long().to(input_ids.device) * span_attention_mask) == 0
        pred_key_padding_mask[:, 0] = False
        act_key_padding_mask = (span_labels != 0).bool().to(input_ids.device)

        key_padding_mask = pred_key_padding_mask if use_pred_attn_mask else act_key_padding_mask

        """ Self MHA between each Fact encoding reduced representation & itself """
        for mha in self.self_mha:
            self_mha_op, _ = mha(query=enc, key=enc, value=enc, key_padding_mask=key_padding_mask)  # batch, seq, h_dim
            enc = enc + self.self_mha_drop(self_mha_op)

        enc_pooled = enc[:, 0, :]
        scheme_pred = self.scheme_prediction_head(enc_pooled)  # batch, n_scheme_classes

        if scheme_labels is not None:
            bce_loss = nn.BCEWithLogitsLoss()
            scheme_loss = bce_loss(scheme_pred, scheme_labels)
            loss += scheme_loss

        op = {"span_logits": span_pred, "scheme_logits": scheme_pred, "loss": loss}

        if get_representation:
            op["representations"] = enc_pooled
        return op

    def predict(self, input_ids, get_representation=False):
        span_attention_mask = (input_ids != self.configuration["tokenizer"].pad_token_id).long().to(input_ids.device)
        enc = self.encoder(input_ids=input_ids, attention_mask=span_attention_mask)  # batch, seq, h_dim
        span_pred = self.span_prediction_head(enc)  # batch, seq, n_classes

        key_padding_mask = ((span_pred.argmax(-1) == 0).long().to(input_ids.device) * span_attention_mask) == 0
        key_padding_mask[:, 0] = False

        """ Self MHA between each Fact encoding reduced representation & itself """
        for mha in self.self_mha:
            self_mha_op, _ = mha(query=enc, key=enc, value=enc, key_padding_mask=key_padding_mask)  # batch, seq, h_dim
            enc = enc + self.self_mha_drop(self_mha_op)

        enc_pooled = enc[:, 0, :]
        scheme_pred = self.scheme_prediction_head(enc_pooled)  # batch, n_scheme_classes

        op = {"span_logits": span_pred, "scheme_logits": scheme_pred, "loss": None}

        if get_representation:
            op["representations"] = enc_pooled
        return op


class DecoderOnlyArgGenerator_M3(nn.Module):
    def __init__(self, base_model1, configuration):
        super().__init__()
        self.configuration = configuration
        self.base_model1 = base_model1
        self.pat_token = self.configuration['tokenizer'].get_vocab()["<pattern>"]
        self.arg_token = self.configuration['tokenizer'].get_vocab()["<argument>"]
        self.pad_token_id = self.configuration['tokenizer'].pad_token_id
        self.var_ids = [self.configuration['tokenizer'].get_vocab()["<" + i + ">"] for i in mappings.variable_list]

    def forward(self, input_dict):
        op1 = self.base_model1(input_ids=input_dict["decoder1_input_mat"],
                               attention_mask=input_dict["decoder1_attn_mat"],
                               labels=input_dict["decoder1_labels_mat"]
                               # labels=None
                               )

        return {"model1_loss": op1["loss"], "model1_logits": op1["logits"], "loss": op1["loss"]}

    def predict(self, input_dict, config_dict={}):
        example_lens = (input_dict["decoder1_attn_mat"] == 1).sum(-1).tolist()
        lst, attn = [], []
        for ix, i in enumerate(input_dict["decoder1_input_mat"]):
            pat_token_idx = [ix2 for ix2, i2 in enumerate(i[-example_lens[ix]:].tolist())
                             if i2 == self.pat_token][0] + 1
            t = i[-example_lens[ix]:].tolist()[:pat_token_idx]
            lst.append(t)
            attn.append([1] * len(t))
        model1_dec_input_ids = torch.tensor(utils.pad_sequence(self.pad_token_id, lst, left=True)).to(
            input_dict["decoder1_input_mat"].device)
        model1_dec_attn_mask = torch.tensor(utils.pad_sequence(0, attn, left=True)).to(model1_dec_input_ids.device)
        assert model1_dec_input_ids.shape == model1_dec_attn_mask.shape

        res = self.base_model1.generate(input_ids=model1_dec_input_ids, attention_mask=model1_dec_attn_mask,
                                        do_sample=config_dict.get("do_sample", False),
                                        early_stopping=config_dict.get("early_stopping", True),
                                        no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                        num_beams=config_dict.get("num_beams", 5),
                                        num_beam_groups=config_dict.get("num_beam_groups", 1),
                                        top_k=50, top_p=0.95,
                                        return_dict_in_generate=True,
                                        num_return_sequences=config_dict.get("num_return_sequences", 1),
                                        max_length=config_dict.get("max_length",
                                                                   model1_dec_input_ids.shape[-1] + 50),
                                        min_length=config_dict.get("min_length",
                                                                   model1_dec_input_ids.shape[-1] + 10))
        responses = [[j for j in i.tolist() if j != self.pad_token_id] for ix, i in enumerate(res["sequences"])]
        """
        Responses format:
        Topic <VAR_0> Text... <VAR_1> Text...<CTRL codes> <pattern> Text.. <VAR_1>...<VAR_0>..<argument> Text...
        """
        return self.configuration['tokenizer'].batch_decode(responses)


class EncoderDecoderArgGenerator_M3(nn.Module):
    def __init__(self, base_model1, base_model2=None, configuration=None):
        super().__init__()
        self.configuration = configuration
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.pat_token = self.configuration['tokenizer'].get_vocab()["<pattern>"]
        self.arg_token = self.configuration['tokenizer'].get_vocab()["<argument>"]
        self.pad_token_id = self.configuration['tokenizer'].pad_token_id
        self.var_ids = [self.configuration['tokenizer'].get_vocab()["<" + i + ">"] for i in mappings.variable_list]

    def forward(self, input_dict):
        teacher_force = random.random() < self.configuration["experiment_details"].get("teacher_force", 1.0)
        op2, pat_token_idx = None, -1
        decoder_inputs = input_dict["decoder1_input_mat"]
        pattern_weight = self.configuration["experiment_details"].get("pattern_weight", None)

        if teacher_force:
            if pattern_weight is None:
                op1 = self.base_model1(input_ids=input_dict["encoder1_input_mat"],
                                       attention_mask=input_dict["encoder1_attn_mat"],
                                       decoder_input_ids=input_dict["decoder1_input_mat"],
                                       decoder_attention_mask=input_dict["decoder1_attn_mat"],
                                       labels=input_dict["decoder1_labels_mat"])
            else:
                weight = torch.ones_like(input_dict["decoder1_labels_mat"])
                for example_ix, example in enumerate(input_dict["decoder1_labels_mat"]):
                    pat_st = 3
                    pat_en = [ix for ix, tok in enumerate(example)
                              if tok == self.configuration['tokenizer'].eos_token_id][0]
                    weight[example_ix][pat_st: pat_en] = pattern_weight
                weight = weight/pattern_weight
                op1 = self.base_model1(input_ids=input_dict["encoder1_input_mat"],
                                       attention_mask=input_dict["encoder1_attn_mat"],
                                       decoder_input_ids=input_dict["decoder1_input_mat"],
                                       decoder_attention_mask=input_dict["decoder1_attn_mat"])
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                masked_lm_loss = loss_fct(op1["logits"].view(-1, op1["logits"].shape[-1]),
                                          input_dict["decoder1_labels_mat"].view(-1))
                op1["loss"] = (masked_lm_loss * weight.view(-1)).mean()
            op = {"model1_loss": op1["loss"], "model1_logits": op1["logits"]}
        else:
            pat_token_idx = [ix for ix, i in enumerate(input_dict["decoder1_input_mat"][0].tolist())
                             if i == self.pat_token][0] + 1
            labels = input_dict["decoder1_labels_mat"][:, pat_token_idx].unsqueeze(-1).contiguous()

            op1 = self.base_model1(input_ids=input_dict["encoder1_input_mat"],
                                   attention_mask=input_dict["encoder1_attn_mat"],
                                   decoder_input_ids=input_dict["decoder1_input_mat"][:, :pat_token_idx],
                                   decoder_attention_mask=input_dict["decoder1_attn_mat"][:, :pat_token_idx])
            past_key_values = op1["past_key_values"]
            loss_fct = nn.CrossEntropyLoss()
            loss_lst = [loss_fct(op1["logits"][:, -1:].view(-1, op1["logits"].shape[-1]), labels.view(-1))]
            encoder_outputs = tuple(op1["encoder_last_hidden_state"])
            decoder_inputs = [op1["logits"][:, -1:].topk(1).indices.detach().squeeze(-1)]
            logit_list = [op1["logits"][:, -1:]]

            for di in range(pat_token_idx, input_dict["decoder1_labels_mat"].shape[1]):
                labels = input_dict["decoder1_labels_mat"][:, di].unsqueeze(-1).contiguous()
                if (labels != -100).sum().item() == 0:
                    break
                op1 = self.base_model1(encoder_outputs=encoder_outputs,
                                       decoder_input_ids=decoder_inputs[-1],
                                       past_key_values=past_key_values)
                past_key_values = op1["past_key_values"]
                loss_lst.append(loss_fct(op1["logits"][:, -1:].view(-1, op1["logits"].shape[-1]), labels.view(-1)))
                decoder_inputs.append(op1["logits"].topk(1).indices.detach().squeeze(-1))
                logit_list.append(op1["logits"][:, -1:])

            loss = sum(loss_lst) / len(loss_lst) if len(loss_lst) > 0 else 0.0
            decoder_inputs = torch.cat(decoder_inputs, -1)
            op = {"model1_loss": loss, "model1_logits": torch.cat(logit_list, 1)}

        if self.configuration["model_type"] == "arg_gen_M3_V2" and \
                input_dict.get("encoder2_input_mat", None) is not None:
            if not teacher_force:
                lst = []
                for ix, i in enumerate(input_dict["encoder1_input_mat"]):
                    enc1_inp = i[:(i != self.pad_token_id).sum().item()].tolist()
                    dec1_ctrl = input_dict["decoder1_input_mat"][ix][1: pat_token_idx].tolist()
                    dec1_pred = []
                    for tok in decoder_inputs[ix].tolist():
                        if tok in [self.configuration['tokenizer'].eos_token_id, self.pad_token_id]:
                            break
                        else:
                            dec1_pred.append(tok)
                    inp_cat = enc1_inp + dec1_ctrl + dec1_pred + [self.configuration['tokenizer'].eos_token_id]
                    lst.append(inp_cat)

                encoder2_input_ids = torch.tensor(utils.pad_sequence(self.pad_token_id, lst)).to(
                    input_dict["encoder1_input_mat"].device)
                encoder2_attention_mask = (encoder2_input_ids != self.pad_token_id).long().to(encoder2_input_ids.device)

            else:
                encoder2_input_ids, encoder2_attention_mask = input_dict["encoder2_input_mat"], \
                                                             input_dict["encoder2_attn_mat"]
            op2 = self.base_model1(input_ids=encoder2_input_ids,  # input_dict["encoder2_input_mat"],
                                   attention_mask=encoder2_attention_mask,  # input_dict["encoder2_attn_mat"],
                                   decoder_input_ids=input_dict["decoder2_input_mat"],
                                   decoder_attention_mask=input_dict["decoder2_attn_mat"],
                                   labels=input_dict["decoder2_labels_mat"])

        elif self.base_model2 is not None:
            op2 = self.base_model2(input_ids=input_dict["encoder2_input_mat"],
                                   attention_mask=input_dict["encoder2_attn_mat"],
                                   decoder_input_ids=input_dict["decoder2_input_mat"],
                                   decoder_attention_mask=input_dict["decoder2_attn_mat"],
                                   labels=input_dict["decoder2_labels_mat"])
        if op2 is not None:
            op["model2_loss"] = op2["loss"]
            op["model2_logits"] = op2["logits"]
            op["loss"] = 0.5 * op["model1_loss"] + 0.5 * op["model2_loss"]

        if op.get("loss", None) is None:
            op["loss"] = op1["loss"]
        return op

    def predict(self, input_dict, config_dict={}):
        model1_enc_input_ids = input_dict["encoder1_input_mat"]
        model1_enc_attn_mask = input_dict["encoder1_attn_mat"]

        pat_token_idx = \
            [ix for ix, i in enumerate(input_dict["decoder1_input_mat"][0].tolist()) if i == self.pat_token][
                0] + 1
        model1_dec_input_ids = input_dict["decoder1_input_mat"][:, :pat_token_idx]
        model1_dec_attn_mask = input_dict["decoder1_attn_mat"][:, :pat_token_idx]

        model1_res = self.base_model1.generate(input_ids=model1_enc_input_ids,
                                               attention_mask=model1_enc_attn_mask,
                                               decoder_input_ids=model1_dec_input_ids,
                                               decoder_attention_mask=model1_dec_attn_mask,
                                               do_sample=config_dict.get("do_sample", False),
                                               early_stopping=config_dict.get("early_stopping", True),
                                               no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                               num_beams=config_dict.get("num_beams", 5),
                                               num_beam_groups=config_dict.get("num_beam_groups", 1),
                                               top_k=50, top_p=0.95,
                                               return_dict_in_generate=True,
                                               num_return_sequences=config_dict.get("num_return_sequences", 1),
                                               max_length=config_dict.get("max_length",
                                                                          model1_dec_input_ids.shape[-1] + 50),
                                               min_length=config_dict.get("min_length",
                                                                          model1_dec_input_ids.shape[-1] + 10))

        """ Removing padding """
        lst = []
        for ix, i in enumerate(input_dict["encoder1_input_mat"]):
            enc1_inp = input_dict["encoder1_input_mat"][ix][:(i != self.pad_token_id).sum().item()]
            mod1_res = model1_res["sequences"][ix]
            tmp_inp = torch.cat([enc1_inp, mod1_res])
            lst.append(tmp_inp[:(tmp_inp != self.pad_token_id).sum().item()].tolist())

        model2_enc_input_ids = torch.tensor(utils.pad_sequence(self.pad_token_id, lst)).to(
            input_dict["encoder1_input_mat"].device)
        model2_enc_attn_mask = (model2_enc_input_ids != self.pad_token_id).long().to(model2_enc_input_ids.device)

        arg_token_idx = \
            [ix for ix, i in enumerate(input_dict["decoder2_input_mat"][0].tolist()) if i == self.arg_token][
                0] + 1
        model2_dec_input_ids = input_dict["decoder2_input_mat"][:, :arg_token_idx]
        model2_dec_attn_mask = input_dict["decoder2_attn_mat"][:, :arg_token_idx]

        model2_res = self.base_model2.generate(input_ids=model2_enc_input_ids,
                                               attention_mask=model2_enc_attn_mask,
                                               decoder_input_ids=model2_dec_input_ids,
                                               decoder_attention_mask=model2_dec_attn_mask,
                                               do_sample=config_dict.get("do_sample", False),
                                               early_stopping=config_dict.get("early_stopping", True),
                                               no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                               num_beams=config_dict.get("num_beams", 5),
                                               num_beam_groups=config_dict.get("num_beam_groups", 1),
                                               top_k=50, top_p=0.95,
                                               return_dict_in_generate=True,
                                               num_return_sequences=config_dict.get("num_return_sequences", 1),
                                               max_length=config_dict.get("max_length",
                                                                          model2_dec_input_ids.shape[-1] + 50),
                                               min_length=config_dict.get("min_length",
                                                                          model2_dec_input_ids.shape[-1] + 10))

        responses = [[j for j in i.tolist() if j != self.pad_token_id] for ix, i in
                     enumerate(torch.cat([model2_enc_input_ids, model2_res["sequences"]], 1))]
        """
        Responses format:
        <s>Topic <VAR_0> Text... <VAR_1> Text... </s>
        <s> <CTRL codes> <pattern> Text.. <VAR_1>...<VAR_0>..</s>
        <s> <CTRL codes> <argument> Text...</s>
        """
        return self.configuration['tokenizer'].batch_decode(responses)

    def predict_single(self, input_dict, config_dict={}):
        enc_input_ids = input_dict["encoder1_input_mat"]
        assert enc_input_ids.shape[0] == 1  # Only 1 example at a time
        enc_attn_mask = (enc_input_ids != self.pad_token_id).long().to(enc_input_ids.device)
        dec_input_ids = input_dict["decoder1_input_mat"]
        dec_attn_mask = (dec_input_ids != self.pad_token_id).long().to(dec_input_ids.device)

        vars_present, vars_absent = [], []
        for i in self.var_ids:
            if i in enc_input_ids:
                vars_present.append([i])
            else:
                vars_absent.append([i])

        res = self.base_model1.generate(input_ids=enc_input_ids,
                                        attention_mask=enc_attn_mask,
                                        decoder_input_ids=dec_input_ids,
                                        decoder_attention_mask=dec_attn_mask,
                                        do_sample=config_dict.get("do_sample", False),
                                        early_stopping=config_dict.get("early_stopping", True),
                                        no_repeat_ngram_size=config_dict.get("no_repeat_ngram_size", 3),
                                        num_beams=config_dict.get("num_beams", 5),
                                        bad_words_ids=vars_absent,
                                        return_dict_in_generate=True,
                                        num_return_sequences=config_dict.get("num_return_sequences", 1),
                                        max_new_tokens=config_dict.get("max_length", 50))
        responses = [[j for j in i.tolist() if j != self.pad_token_id] for ix, i in enumerate(res["sequences"])]
        return self.configuration['tokenizer'].batch_decode(responses)
