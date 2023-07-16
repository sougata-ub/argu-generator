import torch
import argparse
from trainer_models_v2 import Trainer
import time
from tqdm import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import os.path
import random
from datetime import datetime
import nltk
import mappings, utils
from datasets import load_metric
import re

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = "1"
os.environ['NCCL_DEBUG'] = "INFO"

def main():
    N_EPOCHS, batch_size, accumulate, lr, device_num, num_workers, early_stopping = 15, 8, 4, 1e-5, 0, 1, 3
    path_prefix = "" #project path
    skip_training, train_distributed, save_results, save_model = "false", "true", "true", "true"
    best_valid_loss, t_loss, v_loss, best_epoch, test_scores = float('inf'), None, None, None, {}
    pretrained_model = None
    add_special_tokens, run_inference, cross_validation, force_save_model = "true", "false", "false", "false"
    find_unused_parameters = "true"
    n_examples = -1
    rouge = load_metric("rouge")
    pc_model_name = None
    teacher_force = None

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size)
    parser.add_argument("--accumulate", type=int, help="Gradient accumulation steps.", default=accumulate)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=lr)
    parser.add_argument("--device_num", type=int, help="CUDA device number", default=device_num)
    parser.add_argument("--skip_training", type=str, help="Skip Training", default=skip_training)
    parser.add_argument("--train_distributed", type=str, help="train_distributed", default=train_distributed)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=num_workers)
    parser.add_argument("--mname", type=str, help="mname")
    parser.add_argument("--save_results", type=str, help="save_results", default=save_results)
    parser.add_argument("--save_model", type=str, help="save_model", default=save_model)
    parser.add_argument("--best_valid_loss", type=float, help="best_valid_loss", default=best_valid_loss)
    parser.add_argument("--pretrained_model", type=str, help="pretrained_model", default=pretrained_model)
    parser.add_argument("--experiment_number", type=int, help="experiment_number")
    parser.add_argument("--early_stopping", type=int, help="early_stopping", default=early_stopping)
    parser.add_argument("--stats_csv", type=str, help="stats_csv")
    parser.add_argument("--add_special_tokens", type=str, help="add_special_tokens", default=add_special_tokens)
    parser.add_argument("--run_inference", type=str, help="run_inference", default=run_inference)
    parser.add_argument("--model_type", type=str, help="model_type")
    parser.add_argument("--cross_validation", type=str, help="cross_validation")
    parser.add_argument("--span_scheme_test_type", type=str, help="test_type")
    parser.add_argument("--find_unused_parameters", type=str, help="find_unused_parameters",
                        default=find_unused_parameters)
    parser.add_argument("--n_examples", type=int, help="n_examples", default=n_examples)
    parser.add_argument("--force_save_model", type=str, help="force_save_model",
                        default=force_save_model)
    parser.add_argument("--pc_model_name", type=str, help="pc_model_name", default=pc_model_name)
    parser.add_argument("--teacher_force", type=float, help="teacher_force", default=teacher_force)

    argv = parser.parse_args()
    """ Loading config file """
    configurations = {"num_epochs": argv.num_epochs, "batch_size": argv.batch_size, "accumulate": argv.accumulate,
                      "lr": argv.learning_rate,
                      "device_num": int(os.environ["LOCAL_RANK"]),
                      "skip_training": True if argv.skip_training == "true" else False,
                      "train_distributed": True if argv.train_distributed == "true" else False,
                      "num_workers": argv.num_workers, "mname": argv.mname, "early_stopping": argv.early_stopping,
                      "save_results": True if argv.save_results == "true" else False,
                      "save_model": True if argv.save_model == "true" else False,
                      "best_valid_loss": argv.best_valid_loss, "pretrained_model": argv.pretrained_model,
                      "model_type": argv.model_type, "experiment_number": argv.experiment_number,
                      "stats_csv": argv.stats_csv, "span_scheme_test_type": argv.span_scheme_test_type,
                      "add_special_tokens": True if argv.add_special_tokens == "true" else False,
                      "run_inference": True if argv.run_inference == "true" else False,
                      "cross_validation": True if argv.cross_validation == "true" else False,
                      "find_unused_parameters": True if argv.find_unused_parameters == "true" else False,
                      'n_examples': argv.n_examples,
                      "force_save_model": True if argv.force_save_model == "true" else False,
                      "pc_model_name": argv.pc_model_name}

    """ Do not save the CV models. Only save their results"""
    if configurations["cross_validation"]:
        print("The model WON'T be persisted, ONLY the results will be saved as --cross_validation is set to be True!")
        configurations["save_model"] = False
        configurations["save_results"] = True

    if configurations["run_inference"]:
        print("The model and execution details WILL be FORCE SAVED as --run_inference is set to be True!")
        configurations["save_model"] = True
        # configurations["force_save_model"] = True
        configurations["save_results"] = True

    configurations["experiment_details"] = mappings.experiment_map[configurations["model_type"]][
        configurations["experiment_number"]]
    if (configurations["experiment_details"].get("teacher_force", None) is None and argv.teacher_force is not None) or \
            argv.teacher_force is not None:
        configurations["experiment_details"]["teacher_force"] = argv.teacher_force

    if configurations["experiment_details"].get("pretrained_pc_model", None) is not None:
        configurations["pc_model_name"] = configurations["experiment_details"]["pretrained_pc_model"].split("/")[-1]

    if configurations["model_type"] == "arg_gen_M3_V2":
        assert configurations["pc_model_name"] is not None

    keys_dict = pickle.load(open(configurations["experiment_details"]["keys_file"], "rb"))
    if configurations["experiment_details"].get("filter_file", None) is not None:
        filter_keys = pickle.load(open(configurations["experiment_details"]["filter_file"], "rb"))
    else:
        filter_keys = None

    if configurations["model_type"] == "arg_gen_M3":
        k1, k2 = configurations["experiment_details"]["exp_name"].split("_")
        keys_dict = keys_dict[k1][k2]

    """ Creating folder structure """
    result_folder = path_prefix + "<results folder name>"

    if configurations["device_num"] == 0:
        if not os.path.isdir(result_folder) and configurations["save_results"]:
            print("Creating New Directory: ", result_folder)
            os.mkdir(result_folder)
        all_run_nums = [int(i.split("_")[0]) for i in os.listdir(result_folder) if "." not in i and
                        len(re.findall("[0-9]", i)) > 0]  # i.endswith("csv")]
        max_num = 0 if len(all_run_nums) == 0 else max(all_run_nums)
        epoch_time = int(time.time())
        new_folder_name = str(max_num + 1) + "_" + str(epoch_time) + "_" + configurations["model_type"]

        result_folder += new_folder_name + "/"
        destination_result_folder = result_folder + "results/"
        destination_model_folder = result_folder + "models/"
        if not os.path.isdir(result_folder) and configurations["device_num"] == 0 and configurations["save_results"]:
            print("Creating New Directory: ", result_folder)
            os.mkdir(result_folder)
            os.mkdir(destination_result_folder)
            os.mkdir(destination_model_folder)

        configurations["model_name"] = destination_model_folder + "trained_" + configurations[
            "model_type"] + "_model_" + str(max_num + 1) + ".pt"
        configurations["log_name"] = destination_result_folder + "trained_" + configurations[
            "model_type"] + "_model_" + str(max_num + 1) + "_log.pkl"

    torch.distributed.init_process_group(backend="nccl")
    trainer = Trainer(configurations)
    execution_log = {"configuration": configurations}
    stat_df = pd.read_csv(configurations["stats_csv"]) if os.path.exists(
        configurations["stats_csv"]) else pd.DataFrame()

    if not configurations["skip_training"]:
        print("\n:::::: Starting Training ::::::\n")
        if configurations["model_type"] == "fact_span_M1":
            stat_df, execution_log = utils.run_training_fact_span_M1(keys_dict, trainer, configurations, stat_df,
                                                                     best_valid_loss, execution_log,
                                                                     N=configurations["n_examples"],
                                                                     filter_keys=filter_keys)
        elif configurations["model_type"] == "span_scheme_M2":
            stat_df, execution_log = utils.run_training_span_scheme_M2(keys_dict, trainer, configurations, stat_df,
                                                                       best_valid_loss, execution_log,
                                                                       N=configurations["n_examples"],
                                                                       filter_keys=filter_keys)
        elif configurations["model_type"] == "arg_gen_M3_V3":
            stat_df, execution_log = utils.run_training_arg_gen_M3_V3(keys_dict, trainer, configurations, stat_df,
                                                                      best_valid_loss, execution_log,
                                                                      N=configurations["n_examples"],
                                                                      filter_keys=filter_keys)
        else:
            print("INVALID CHOICE!!")
    if configurations["device_num"] == 0:
        if configurations["run_inference"]:
            print("\nLoading best weights for PC model from:", configurations["model_name"], "\n")
            state_dict = torch.load(configurations["model_name"])
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            trainer.model.load_state_dict(state_dict)
            trainer.model.eval()
            pc_test_scores = trainer.generate_response_V2(keys_dict["pc_basn"][0]["test"]
                                                          [: configurations["n_examples"]], rouge)

            execution_log["pc_test_scores"], best_val_pc = pc_test_scores, 0.0

            try:
                best_val_pc = 0.0 if execution_log["pc_basn"][0].get("best_valid_loss", None) is None else execution_log["pc_basn"][0]["best_valid_loss"]
            except Exception as e:
                print("Exception:", e)
                print("execution_log[pc_basn]:", execution_log["pc_basn"][0].keys())

            try:
                tmpdf = pd.DataFrame([[configurations["experiment_number"]] + pc_test_scores.get("test_scores_pat", {}).get("summary", [None, None, None, None]) + [round(math.exp(best_val_pc), 4)],
                                      [configurations["experiment_number"]] + pc_test_scores.get("test_scores_arg", {}).get("summary", [None, None, None, None]) + [round(math.exp(best_val_pc), 4)]],
                                     columns=["exp_id", "metadata", "bleu_score", "bleu_4", "rougeL", "valid_PPL"])
                stat_df = pd.concat([stat_df, tmpdf]).reset_index(drop=True)
            except Exception as e:
                print("Exception:", e)

        if configurations["save_results"]:
            print("Saving Execution to Log File:", configurations["log_name"])
            pickle.dump(execution_log, open(configurations["log_name"], "wb"))
            if stat_df.shape[0] > 0:
                stat_df.to_csv(configurations["stats_csv"], index=False)
                print("Saving stats to csv file:", configurations["stats_csv"])
            else:
                print("NOT Saving stats to csv file, as it is EMPTY!")

        if not configurations["save_model"] and not configurations["skip_training"]:
            if os.path.exists(configurations["model_name"]):
                os.remove(configurations["model_name"])
                print("DELETED MODEL:", configurations["model_name"], "!")
            else:
                print("COULD NOT FIND ANY MODEL TO DELETE IN THE SPECIFIED PATH!!!!")

        print("\nALL DONE AT TIMESTAMP:", int(time.time()), "!! GOING TO EXIT!!\n")
        os._exit(1)
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
