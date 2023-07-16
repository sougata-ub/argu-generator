# ArgU: A Controllable Factual Argument Generator
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of the paper:

## [**ArgU: A Controllable Factual Argument Generator**]([https://aclanthology.org/2022.nlp4convai-1.16/](https://aclanthology.org/2023.acl-long.466.pdf)). 
[**Sougata Saha**](https://sougata-ub.github.io), [**Rohini Srihari**](https://www.acsu.buffalo.edu/~rohini/) 

Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. Volume 1: Long Papers, pages 8373–8388. July 9-14, 2023 (ACL 2023)

## Abstract
Effective argumentation is essential towards a purposeful conversation with a satisfactory out- come. For example, persuading someone to reconsider smoking might involve empathetic, well founded arguments based on facts and expert opinions about its ill-effects and the consequences on one’s family. However, the automatic generation of high-quality factual arguments can be challenging. Addressing existing controllability issues can make the recent advances in computational models for argument generation a potential solution. In this paper, we introduce ArgU: a neural argument generator capable of producing factual arguments from input facts and real-world concepts that can be explicitly controlled for stance and argument structure using Walton’s argument scheme-based control codes. Unfortunately, computational argument generation is a relatively new field and lacks datasets conducive to training. Hence, we have compiled and released an annotated corpora of 69,428 arguments spanning six topics and six argument schemes, making it the largest publicly available corpus for identifying argument schemes; the paper details our annotation and dataset creation framework. We further experiment with an argument generation strategy that establishes an inference strategy by generating an “argument template” before actual argument generation. Our results demonstrate that it is possible to automatically generate diverse arguments exhibiting different inference patterns for the same set of facts by using control codes based on argument schemes and stance.

### Data:
1. ArgSpan and ArgSpanScheme data: https://argu-files.s3.amazonaws.com/arg_span_and_scheme_data.pkl
2. ArgSpan and ArgSpanScheme training and testing splits: https://argu-files.s3.amazonaws.com/arg_span_and_scheme_data_keys.pkl
3. ArgU data: https://argu-files.s3.amazonaws.com/argu_generator_data.pkl
4. ArgU training and testing splits: https://argu-files.s3.amazonaws.com/argu_generator_keys.pkl
   
### Models and Configuration Files:
1. ArgU Mono model: https://argu-files.s3.amazonaws.com/argu_mono_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_mono_generator_log.pkl
2. ArgU Dual model: https://argu-files.s3.amazonaws.com/argu_dual_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_dual_generator_log.pkl
3. ArgU Stance model: https://argu-files.s3.amazonaws.com/argu_stance_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_stance_generator_log.pkl
4. ArgU Scheme model: https://argu-files.s3.amazonaws.com/argu_scheme_generator_model.pt, log file: https://argu-files.s3.amazonaws.com/argu_scheme_generator_log.pkl

To perform inference just load the model's log file and weights

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
