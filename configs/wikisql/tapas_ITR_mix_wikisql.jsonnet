{
  DATA_FOLDER: '',
  EXPERIMENT_FOLDER: '',
  TENSORBOARD_FOLDER: '',
  WANDB: {
    CACHE_DIR: '',
    DIR: '',
    entity: 'observer-wandb-seoul-national-university',
    project: 'TableQA_publication',
    tags: [],
  },
  cache: {
    default_folder: 'cache',
    regenerate: {},
  },
  cuda: 0,
  data_loader: {
    additional: {
      max_decoder_source_length: 512,
      max_source_length: 512,
      max_target_length: 128,
      num_knowledge_passages: 5,
    },
    dataset_modules: {
      module_dict: {
        LoadDataLoaders: {
          config: {
            test: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'test',
                use_column: 'wikisql_data',
              },
            ],
            train: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'train',
                use_column: 'wikisql_data',
              },
            ],
            valid: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'validation',
                use_column: 'wikisql_data',
              },
            ],
          },
          option: 'default',
          type: 'LoadDataLoaders',
        },
        LoadWikiSQLData: {
          config: {
            path: {
              test: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_test.arrow',
              train: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_train.arrow',
              validation: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_validation.arrow',
            },
            preprocess: [
              'use_original_tapas_data',
              'split_table_by_mixed_combination',
            ],
            tapas_path: {
              test: 'TableQA_data/wikisql/tf_examples/test.pkl',
              train: 'TableQA_data/wikisql/tf_examples/train.pkl',
              validation: 'TableQA_data/wikisql/tf_examples/dev.pkl',
            },
          },
          option: 'default',
          type: 'LoadWikiSQLData',
        },
        LoadWikiTQData: {
          config: {
            path: {
              test: 'TableQA_data/wtq/preprocessed_test.arrow',
              train: 'TableQA_data/wtq/preprocessed_train.arrow',
              validation: 'TableQA_data/wtq/preprocessed_validation.arrow',
            },
            preprocess: [],
          },
          option: 'default',
          type: 'LoadWikiTQData',
        },
      },
      module_list: [
        'LoadWikiSQLData',
        'LoadDataLoaders',
      ],
    },
    datasets: {},
    dummy_dataloader: 0,
    type: 'DataLoaderForTableQA',
  },
  experiment_name: 'default_test',
  gpu_device: 0,
  ignore_pretrained_weights: [],
  metrics: [
    {
      name: 'compute_tapas_denotation_accuracy',
    },
    {
      name: 'compute_tapas_denotation_accuracy',
      option: 'valid_samples_only',
    },
    {
      name: 'compute_ITR_RAG_retrieval_results',
    },
  ],
  model_config: {
    DECODER_SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    DecoderTokenizerClass: 'CustomTapasTokenizer',
    DecoderTokenizerModelVersion: 'google/tapas-base',
    GeneratorConfigClass: 'TapasConfig',
    GeneratorConfigPredefinedSet: 'WTQ',
    GeneratorModelClass: 'TapasForQuestionAnswering',
    GeneratorModelVersion: 'google/tapas-base-finetuned-wikisql-supervised',
    Ks: [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      20,
      50,
      80,
      100,
    ],
    ModelClass: 'ITRRagReduceMixModel',
    QueryEncoderConfigClass: 'DPRConfig',
    QueryEncoderModelClass: 'DPRQuestionEncoder',
    QueryEncoderModelVersion: '$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/train/saved_model/step_11604/query_encoder',
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    TokenizerClass: 'DPRQuestionEncoderTokenizer',
    TokenizerModelVersion: 'facebook/dpr-question_encoder-single-nq-base',
    base_model: 'ITR_TAPAS',
    decoder_input_modules: {
      module_list: [],
      postprocess_module_list: [],
    },
    index_files: {
      index_paths: {
        test: 'DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.test',
        validation: 'DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.validation',
      },
    },
    input_modules: {
      module_list: [
        {
          option: 'default',
          separation_tokens: {
            end: '',
            start: '',
          },
          type: 'QuestionInput',
        },
      ],
      postprocess_module_list: [
        {
          option: 'default',
          type: 'PostProcessInputTokenization',
        },
      ],
    },
    loss_ratio: {
      nll_loss: 1,
    },
    min_columns: 2,
    modules: [
      'freeze_question_encoder',
    ],
    num_beams: 5,
    output_modules: {
      module_list: [],
      postprocess_module_list: [],
    },
    pretrained: 1,
    rag_modules: {
      module_list: [],
    },
  },
  platform_type: 'pytorch',
  seed: 2022,
  test: {
    additional: {
      multiprocessing: 4,
    },
    batch_size: 32,
    evaluation_name: 'test_evaluation',
    load_best_model: 0,
    load_epoch: -1,
    load_model_path: '',
    num_evaluation: 0,
  },
  train: {
    adam_epsilon: 1e-08,
    additional: {
      early_stop_patience: 3,
      gradient_accumulation_steps: 4,
      gradient_clipping: 0,
      label_smoothing_factor: 0.10000000000000001,
      plugins: [],
      save_top_k: -1,
      save_top_k_metric: 'valid/ITRRAGWikiSQLDataset.validation/denotation_accuracy',
      save_top_k_mode: 'max',
      warmup_steps: 0,
      weight_decay: 0.01,
    },
    batch_size: 32,
    epochs: 9999,
    load_best_model: 0,
    load_epoch: -1,
    load_model_path: '',
    lr: 0.0001,
    save_interval: 1000,
    scheduler: 'none',
    type: 'ITRRAGExecutorForTAPAS',
  },
  valid: {
    additional: {},
    batch_size: 32,
    break_interval: 3000,
    step_size: 1000,
  },
}
