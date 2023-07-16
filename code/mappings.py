prediction_heads = {"span": {"O": 0, "B-clause": 1, "I-clause": 2},
                    "span-type": {"O": 0, "B-clause": 1, "I-clause": 2,
                                  "B-fact": 3, "I-fact": 4,
                                  "B-person_thing": 5, "I-person_thing": 6,
                                  "B-personal_interpretation": 7, "I-personal_interpretation": 8},
                    "scheme-walton-broad": {'applying rules to cases': 0,
                                            'other': 1,
                                            'reasoning': 2,
                                            'source-based arguments': 3},
                    "basn-reduced": {'from consequence': 0,
                                     'from source authority': 1,
                                     'from source knowledge': 2,
                                     'goal from means': 3,
                                     'means for goal': 4,
                                     'other': 5,
                                     'rule or principle': 6},
                    "basn-all": {'cause to effect': 0,
                                 'comparative constraint': 1,
                                 'consistent constraint': 2,
                                 'contradict constraint': 3,
                                 'correlation to cause': 4,
                                 'either constraint': 5,
                                 'from consequence': 6,
                                 'from evidence': 7,
                                 'from example': 8,
                                 'from source authority': 9,
                                 'from source knowledge': 10,
                                 'goal from means': 11,
                                 'interpretation': 12,
                                 'means for goal': 13,
                                 'rule or principle': 14,
                                 'verbal classification': 15,
                                 'verbal evaluation': 16}
                    }

fact_span_experiments = {0: {'exp_id': 0, "type": "fact_span_M1",
                             'exp_name': 'label_input_fact',
                             "arg_max_len": 97,
                             "threshold": 0.5,
                             "out_dim": 600,
                             "n_classes": 3,
                             "keys_file": "<path to arg_span_and_scheme_data_keys.pkl>",
                             "data_file": "<path to arg_span_and_scheme_data.pkl>"},
                         }
span_scheme_experiments = {0: {'exp_id': 0, "type": "span_scheme_M2", "pipelined": False,
                               'exp_name': 'span_scheme_from_arg', "threshold": 0.5,
                               "n_classes_span": 3, "n_classes_scheme": 6,
                               "keys_file": "<path to arg_span_and_scheme_data_keys.pkl>",
                               "data_file": "<path to arg_span_and_scheme_data.pkl>",
                               'inputs': ['argument'],
                               'outputs': [{'head': 'span', 'head_type': 'multiclass'},
                                           {'head': 'basn-reduced', 'head_type': 'binary'}]},

                           1: {'exp_id': 1, "type": "span_scheme_M2", "pipelined": True,
                               'exp_name': 'span_scheme_from_arg_pipelined', "threshold": 0.5,
                               "n_classes_span": 3, "n_classes_scheme": 6, "n_heads": 4, "n_self_attn_layers": 2,
                               "keys_file": "<path to arg_span_and_scheme_data_keys.pkl>",
                               "data_file": "<path to arg_span_and_scheme_data.pkl>",
                               'inputs': ['argument'],
                               'outputs': [{'head': 'span', 'head_type': 'multiclass'},
                                           {'head': 'basn-reduced', 'head_type': 'binary'}]}}

arg_gen_experiments_v3 = {0: {"exp_id": 0, "type": "arg_gen_M3_V3", "exp_name": "fact_stance_scheme_to_arg",
                              "base_model": "facebook/bart-base", "multi_encoder": False, "use_pc": True,
                              "control": "stance_scheme", "target": "argument", "multi_model": False,
                              "keys_file": "<path to argu_generator_keys.pkl>",
                              "data_file": "<path to argu_generator_data.pkl>",
                              },

                          1: {"exp_id": 1, "type": "arg_gen_M3_V3", "exp_name": "fact_stance_scheme_to_pattern_to_arg",
                              "base_model": "facebook/bart-base", "multi_encoder": False, "use_pc": True,
                              "control": "stance_scheme", "target": "argument", "multi_model": False,
                              "keys_file": "<path to argu_generator_keys.pkl>",
                              "data_file": "<path to argu_generator_data.pkl>",
                              },
                          2: {"exp_id": 2, "type": "arg_gen_M3_V3", "exp_name": "fact_stance_to_arg",
                              "base_model": "facebook/bart-base", "multi_encoder": False, "use_pc": True,
                              "control": "stance", "target": "argument", "multi_model": False,
                              "keys_file": "<path to argu_generator_keys.pkl>",
                              "data_file": "<path to argu_generator_data.pkl>",
                              },
                          3: {"exp_id": 3, "type": "arg_gen_M3_V3", "exp_name": "fact_scheme_to_arg",
                              "base_model": "facebook/bart-base", "multi_encoder": False, "use_pc": True,
                              "control": "scheme", "target": "argument", "multi_model": False,
                              "keys_file": "<path to argu_generator_keys.pkl>",
                              "data_file": "<path to argu_generator_data.pkl>",
                              },
                          }


experiment_map = {"fact_span_M1": fact_span_experiments,
                  "span_scheme_M2": span_scheme_experiments,
                  "arg_gen_M3_V3": arg_gen_experiments_v3}

generator_token_map = {'pro': '<pro>',
                       'con': '<con>',
                       'from source knowledge': '<from_source_knowledge>',
                       'from source authority': '<from_source_authority>',
                       'means for goal': '<means_for_goal>',
                       'rule or principle': '<rule_or_principle>',
                       'from consequence': '<from_consequence>',
                       'reasoning': '<reasoning>',
                       'applying rules to cases': '<applying_rules_to_cases>',
                       'source-based arguments': '<source_based_arguments>'}

generator_token_map2 = {'pro': '<pro>',
                        'con': '<con>',
                        'from source knowledge': '<from_source_knowledge>',
                        'from source authority': '<from_source_authority>',
                        'means for goal': '<means_for_goal>',
                        'goal from means': '<means_for_goal>',
                        'rule or principle': '<rule_or_principle>',
                        'from consequence': '<from_consequence>'}

scheme_to_id = {'cause to effect': 0,
                'comparative constraint': 1,
                'consistent constraint': 2,
                'contradict constraint': 3,
                'correlation to cause': 4,
                'either constraint': 5,
                'from consequence': 6,
                'from evidence': 7,
                'from example': 8,
                'from source authority': 9,
                'from source knowledge': 10,
                'goal from means': 11,
                'interpretation': 12,
                'means for goal': 13,
                'rule or principle': 14,
                'verbal classification': 15,
                'verbal evaluation': 16}

walton_scheme_to_id = {'applying rules to cases': 0,
                       'other': 1,
                       'reasoning': 2,
                       'source-based arguments': 3}

filtered_scheme_to_id = {'from consequence': 0,
                         'from source authority': 1,
                         'from source knowledge': 2,
                         'goal from means': 3,
                         'means for goal': 4,
                         'other': 5,
                         'rule or principle': 6}

new_scheme_to_id = {'from consequence': 0, 'from source authority': 1, 'from source knowledge': 2,
                    'goal from means/means for goal': 3, 'other': 4, 'rule or principle': 5}

variable_list = ["VAR_" + str(i) for i in range(4)]
misc_list = ["pattern", "argument", "pro", "con"]
generator_token_M3 = {i: "<" + "_".join(i.split()) + ">" for i in
                      list(new_scheme_to_id.keys()) + variable_list + misc_list
                      if i != "other"}

id_to_scheme = {ix: i for i, ix in scheme_to_id.items()}
id_to_scheme_walton = {ix: i for i, ix in walton_scheme_to_id.items()}
id_to_scheme_filtered = {ix: i for i, ix in filtered_scheme_to_id.items()}
id_to_scheme_new = {ix: i for i, ix in new_scheme_to_id.items()}

id_to_span = {ix: i for i, ix in prediction_heads["span"].items()}
id_to_span_type = {ix: i for i, ix in prediction_heads["span-type"].items()}
