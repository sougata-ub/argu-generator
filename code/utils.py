import torch
import time
from tqdm import tqdm
import copy
import math
import pandas as pd
import os
import torch.distributed as dist
import random


def pad_sequence(pad, batch, left=False):
    maxlen = max([len(i) for i in batch])
    lst = []
    for i in batch:
        if left:
            lst.append([pad] * (maxlen - len(i)) + i)
        else:
            lst.append(i + [pad] * (maxlen - len(i)))
    return lst


def get_splits_to_run(keys_dict, test_type):
    # test_type -> topic:0,3#random:1
    dct = {}
    for typ_phrase in test_type.split("#"):
        if len(typ_phrase.split(":")) > 1:
            typ, idx_str = typ_phrase.split(":")
            for idx in idx_str.split(","):
                ky = typ + ":" + idx
                dct[ky] = keys_dict[typ][int(idx)]
        else:
            for k, v in keys_dict[typ_phrase].items():
                dct[typ_phrase + ":" + str(k)] = v
    return dct


def run_training(keys_dct, trainer, configuration, best_valid_loss, cv_id=None, N=-1, filter_keys=None):
    t_loss, t_loss_m1, t_loss_m2, v_loss, v_loss_m1, v_loss_m2 = [], [], [], [], [], []

    best_epoch = 0
    early_stopping_marker = []
    best_stats = None

    train_dl, test_dl = trainer.get_dataloaders(keys_dct, N=N, filter_keys=filter_keys)

    for epoch in range(configuration["num_epochs"]):
        if cv_id is not None:
            print("CV: {}, Epoch: {}, Training ...\n".format(cv_id, epoch))
        else:
            print("Epoch: {}, Training ...\n".format(epoch))
        start_time = time.time()

        tr_l, tr_l_m1, tr_l_m2 = trainer.train(train_dl)
        t_loss.append(tr_l)
        t_loss_m1.append(tr_l_m1)
        t_loss_m2.append(tr_l_m2)

        if configuration["device_num"] == 0:
            print("Epoch: {}, Evaluating ...\n".format(epoch))

            vl_l, vl_l_m1, vl_l_m2, stats = trainer.evaluate(test_dl)
            v_loss.append(vl_l)
            v_loss_m1.append(vl_l_m1)
            v_loss_m2.append(vl_l_m2)
            end_time = time.time()
            epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

            if vl_l <= best_valid_loss:
                best_valid_loss = vl_l
                print("FOUND BEST MODEL!")
                if configuration["save_model"]:
                    print("SAVING BEST MODEL!")
                    torch.save(trainer.ddp_model.state_dict(), configuration["model_name"])
                    print("BEST MODEL SAVED to location ", configuration["model_name"], "!\n")
                    if configuration["model_type"] == "arg_gen_M3_V2" and cv_id is None:
                        torch.save(trainer.ddp_model.state_dict(), configuration["temp_model_name"])
                        print("BEST MODEL ALSO SAVED to TEMP location ", configuration["temp_model_name"], "!\n")
                best_epoch = epoch
                best_stats = stats
                early_stopping_marker.append(False)
            else:
                if configuration["force_save_model"]:
                    fs_mname = configuration["model_name"].replace(".pt", "_ckpt_" + str(epoch) + ".pt")
                    print("FORCE SAVING MODEL TO LOCATION:", fs_mname, "!")
                    torch.save(trainer.ddp_model.state_dict(), fs_mname)
                early_stopping_marker.append(True)
            print("\n")
            if cv_id is not None:
                print(f'CV: {cv_id} | Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
            else:
                print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Total Loss: {tr_l:.3f} | Val Total Loss: {vl_l:.3f}')

            if trainer.configuration["model_type"].startswith("arg_gen_M3"):
                print(f'\tTrain Decoder1 Loss: {tr_l_m1:.3f} | Val Decoder1 Loss: {vl_l_m1:.3f} | '
                      f'Train Decoder1 PPL: {math.exp(tr_l_m1):.3f} | Val Decoder1 PPL: {math.exp(vl_l_m1):.3f}')
                if 'bart' in trainer.configuration["experiment_details"]["base_model"] and \
                        trainer.configuration["experiment_details"].get("multi_encoder", False):
                    print(f'\tTrain Decoder2 Loss: {tr_l_m2:.3f} | Val Decoder2 Loss: {vl_l_m2:.3f} | '
                          f'Train Decoder2 PPL: {math.exp(tr_l_m2):.3f} | Val Decoder2 PPL: {math.exp(vl_l_m2):.3f}')
            else:
                if "generator" in trainer.configuration["model_type"]:
                    print(f'\tTrain PPL: {math.exp(tr_l):.3f} | Val PPL: {math.exp(vl_l):.3f}')

            if all(early_stopping_marker[-configuration["early_stopping"]:]) and \
                    len(early_stopping_marker) >= configuration["early_stopping"]:
                print("Early stopping training as the Validation loss did NOT improve for last " + str(
                    configuration["early_stopping"]) + \
                      " iterations.")
                break
        # dist.barrier()
    return t_loss, v_loss, t_loss_m1, v_loss_m1, t_loss_m2, v_loss_m2, best_valid_loss, best_epoch, best_stats


def run_training_fact_span_M1(keys_dct, trainer, configurations, stat_df, best_valid_loss=None, execution_log=None,
                              N=-1, filter_keys=None):
    if execution_log is None:
        execution_log = {"configuration": configurations}

    t_loss, v_loss, _, _, _, _, best_valid_loss, best_epoch, best_stats = run_training(keys_dct, trainer,
                                                                                       configurations, best_valid_loss,
                                                                                       N=N, filter_keys=filter_keys)

    if configurations["device_num"] == 0:
        tmpdf = pd.DataFrame(best_stats["span_stats"], columns=["Task", "Label", "Metric", "Score"])
        tmpdf["experiment_number"] = configurations["experiment_number"]
        tmpdf["model_type"] = configurations["model_type"]
        stat_df = pd.concat([stat_df, tmpdf]).reset_index(drop=True)
        # stat_df.to_csv(configurations["stats_csv"], index=False)
        # print("Saving stats to csv file:", configurations["stats_csv"])
        execution_log = {**execution_log,
                         **{"t_loss": t_loss, "v_loss": v_loss, "best_valid_loss": best_valid_loss,
                            "best_epoch": best_epoch, "best_stats": best_stats}
                         }

    return stat_df, execution_log


def run_training_span_scheme_M2(keys_dct, trainer, configurations, stat_df, best_valid_loss=None, execution_log=None,
                                N=-1, filter_keys=None):
    splits_to_run = get_splits_to_run(keys_dct, configurations["span_scheme_test_type"])
    if execution_log is None:
        execution_log = {"configuration": configurations}

    for cv_id, v in splits_to_run.items():
        best_valid_loss = float('inf')  # if best_valid_loss is None else best_valid_loss
        if configurations["device_num"] == 0 and configurations["cross_validation"]:
            trainer.reset_model()
        t_loss, v_loss, _, _, _, _, best_valid_loss, best_epoch, best_stats = run_training(v, trainer,
                                                                                           configurations,
                                                                                           best_valid_loss, cv_id=cv_id,
                                                                                           N=N,
                                                                                           filter_keys=filter_keys)

        if configurations["device_num"] == 0:
            execution_log[cv_id] = {"t_loss": t_loss, "v_loss": v_loss, "best_valid_loss": best_valid_loss,
                                    "best_epoch": best_epoch, "best_stats": best_stats}
            tmpdf = pd.DataFrame()
            if best_stats["scheme_stats_binary"] is not None:
                tmpdf = pd.concat([tmpdf,
                                   pd.DataFrame(best_stats["scheme_stats_binary"],
                                                columns=["Task", "Label", "Metric", "Score"])])
            if best_stats["scheme_stats_by_class"] is not None:
                tmpdf = pd.concat([tmpdf,
                                   pd.DataFrame(best_stats["scheme_stats_by_class"],
                                                columns=["Task", "Label", "Metric", "Score"])])
            if best_stats["span_stats"] is not None:
                tmpdf = pd.concat([tmpdf,
                                   pd.DataFrame(best_stats["span_stats"],
                                                columns=["Task", "Label", "Metric", "Score"])])
            if best_stats["span_eval"] is not None:
                tmpdf = pd.concat([tmpdf,
                                   pd.DataFrame(best_stats["span_eval"]["overlap_result"],
                                                columns=["Task", "Label", "Metric", "Score"])])
            tmpdf["CV"] = cv_id
            print("Consolidated Stats\n")
            print(tmpdf)
            tmpdf["experiment_number"] = configurations["experiment_number"]
            tmpdf["test_type"] = configurations["span_scheme_test_type"]
            tmpdf["model_type"] = configurations["model_type"]
            # tmpdf["folder"] = new_folder_name
            stat_df = pd.concat([stat_df, tmpdf])
            # stat_df.to_csv(configurations["stats_csv"], index=False)
            # print("Saving stats to csv file:", configurations["stats_csv"])

    return stat_df, execution_log


def run_training_arg_gen_M3_V3(keys_dct, trainer, configurations, stat_df, best_valid_loss=None, execution_log=None,
                               N=-1, filter_keys=None):
    if execution_log is None:
        execution_log = {"configuration": configurations}

    t_loss, v_loss, t_loss_m1, v_loss_m1, t_loss_m2, \
    v_loss_m2, best_valid_loss, best_epoch, best_stats = run_training(keys_dct["pc_basn"][0], trainer, configurations,
                                                                      best_valid_loss, N=N, filter_keys=filter_keys)
    if configurations["device_num"] == 0:
        execution_log["pc_basn"] = {
            0: {"t_loss": t_loss, "v_loss": v_loss, "t_loss_m1": t_loss_m1, "v_loss_m1": v_loss_m1,
                "t_loss_m2": t_loss_m2, "v_loss_m2": v_loss_m2, "best_valid_loss": best_valid_loss,
                "best_epoch": best_epoch, "best_stats": best_stats}
            }
    return stat_df, execution_log
