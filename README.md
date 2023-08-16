# ArgU: A Controllable Factual Argument Generator
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of the paper:

## [**ArgU: A Controllable Factual Argument Generator**](https://aclanthology.org/2023.acl-long.466.pdf). 
[**Sougata Saha**](https://sougata-ub.github.io), [**Rohini Srihari**](https://www.acsu.buffalo.edu/~rohini/) 

Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. Volume 1: Long Papers, pages 8373–8388. July 9-14, 2023 (ACL 2023)

## Abstract
Effective argumentation is essential towards a purposeful conversation with a satisfactory out- come. For example, persuading someone to reconsider smoking might involve empathetic, well founded arguments based on facts and expert opinions about its ill-effects and the consequences on one’s family. However, the automatic generation of high-quality factual arguments can be challenging. Addressing existing controllability issues can make the recent advances in computational models for argument generation a potential solution. In this paper, we introduce ArgU: a neural argument generator capable of producing factual arguments from input facts and real-world concepts that can be explicitly controlled for stance and argument structure using Walton’s argument scheme-based control codes. Unfortunately, computational argument generation is a relatively new field and lacks datasets conducive to training. Hence, we have compiled and released an annotated corpora of 69,428 arguments spanning six topics and six argument schemes, making it the largest publicly available corpus for identifying argument schemes; the paper details our annotation and dataset creation framework. We further experiment with an argument generation strategy that establishes an inference strategy by generating an “argument template” before actual argument generation. Our results demonstrate that it is possible to automatically generate diverse arguments exhibiting different inference patterns for the same set of facts by using control codes based on argument schemes and stance.

### Data:
1. ArgSpan and ArgSpanScheme data: https://argu-files.s3.amazonaws.com/arg_span_and_scheme_data.pkl
2. ArgSpan and ArgSpanScheme training and testing splits: https://argu-files.s3.amazonaws.com/arg_span_and_scheme_data_keys.pkl
3. ArgU data: https://argu-files.s3.amazonaws.com/argu_generator_data.pkl
4. ArgU training and testing splits: https://argu-files.s3.amazonaws.com/argu_generator_keys.pkl

### ArgSpan and ArgSpanScheme Data Dictionary
- id: Unique id.
- text: Argument text.
- Statement_[0 to 6]: Available Facts/Concepts.
- label: Span annotation.
- label_type: Span annotation golden or silver.
- stance: pro or con stance.
- idiom: Argument schemes across 7 classes
- lbl_bio: BIO labels of Argument Text.
- model_1_format: Example in ArgSpan training format.
  - text: Roberta formatted argument text.
  - fact: Roberta formatted argument facts. 
  - labels: BIO labels for each input fact. The labels are assigned -1 if input fact is absent. 
  - text_ids: Roberta encoded argument text. 
  - fact_ids: Roberta encoded facts. 
  - tag:
    - train: golden annotated sample used for training.
    - test: golden annotated sample used for testing. 
    - inference: model predicted labels (silver annotation).
- model_2_format: Example in ArgSpanScheme training format.
  - text_input_ids: Roberta formatted argument text. 
  - bio_labels: BIO span labels. 
  - lbl_idiom_new: Argument schemes across 6 classes: 0: 'from consequence', 1: 'from source authority', 2: 'from source knowledge', 3: 'goal from means/means for goal', 4: 'other', 5: 'rule or principle'. 

### ArgU Generator Data Dictionary
- index: unique id.
- topic: topic of the argument.
- stance: pro/con stace of the argument.
- sent: argument text.
- dataset: source dataset.
- basn_lbl: argument scheme of the argument. One of 'goal from means/means for goal', 'from consequence', 'rule or principle', 'from source authority', 'from source knowledge'.
- clean_sent: cleansed and normalized argument text.  
- sent_var: the target template from the argument text.
- var_map: the variable to argument span (fact/concept) mapping for the template.
- span_map_rev: mapping from the identified fact/concept argument span to the normalized set of facts/concepts from the entire corpus.

- model_3_format
  - in_dict: relevant input data fields used during model training.
  - out_dict: relevant output data fields used during model training.
  - ctrl: control codes used during model training.
  - V2_[0 to 3]: encoder and decoder ids used by each model architecture: 0:mono, 1:dual, 2:only stance, 3: only scheme.


### Models and Configuration Files:
1. ArgU Mono model: https://argu-files.s3.amazonaws.com/argu_mono_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_mono_generator_log.pkl
2. ArgU Dual model: https://argu-files.s3.amazonaws.com/argu_dual_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_dual_generator_log.pkl
3. ArgU Stance model: https://argu-files.s3.amazonaws.com/argu_stance_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_stance_generator_log.pkl
4. ArgU Scheme model: https://argu-files.s3.amazonaws.com/argu_scheme_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_scheme_generator_log.pkl
5. ArgSpan model: https://argu-files.s3.amazonaws.com/arg_span_model.pt, log file: https://argu-files.s3.amazonaws.com/arg_span_log.pkl
6. ArgSpanScheme model: https://argu-files.s3.amazonaws.com/arg_span_scheme_model.pt, log file: https://argu-files.s3.amazonaws.com/arg_span_scheme_log.pkl

### Training and Inference
To perform inference just load the model's log file and weights.
```
configurations = pickle.load(open("<log file>", "rb))["configurations"]
configurations["train_distributed"] = False
trainer = Trainer(configurations)
rouge = load_metric("rouge")

state_dict = torch.load(configurations["model_name"])
state_dict = {k[7:]: v for k, v in state_dict.items()}
trainer.model.load_state_dict(state_dict)
trainer.model.eval()
pc_test_scores = trainer.generate_response_V2(keys_dict["pc_basn"][0]["test"], rouge)
```

To train the models from scratch you can use a command like below. Refer to the respective model's log file for a detailed configuration. Additionally check out the `mappings.py` and `run_training_models_v2.py` files to know more details of each parameter.
```
nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port "9999" ./run_training_models_v2.py --num_workers 2 --batch_size 24 --num_epochs 40 --accumulate 1 --learning_rate 0.00001 --early_stopping 5 --save_results "true" --save_model "true" --stats_csv "<path>/arg_gen_M3_V3_stats_v1.csv" --experiment_number <choose from 0 to 3> --train_distributed "true" --model_type <choose from "fact_span_M1", "span_scheme_M2", "arg_gen_M3_V3"> --mname "facebook/bart-base" --add_special_tokens "true" --find_unused_parameters "false" --run_inference "true" --force_save_model "false" --save_model "true" --pc_model_name "<provide a pre-trained model if any>" > log0.txt 2>&1 &
```
### Contact:
1. Please create and submit an issue if you face any difficulties.
2. Additionally you can contact the repo owner via email: sougatas@buffalo.edu

## Citation
If you are using this library then do cite: 
```bibtex
@inproceedings{saha-srihari-2023-argu,
    title = "{A}rg{U}: A Controllable Factual Argument Generator",
    author = "Saha, Sougata  and
      Srihari, Rohini",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.466",
    pages = "8373--8388"
}
```
