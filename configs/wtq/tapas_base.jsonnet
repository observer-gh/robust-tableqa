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
    regenerate: {
      ocr_feature_preprocessed: 0,
      test_data_preprocessed: 1,
      train_data_preprocessed: 1,
      vinvl_feature_preprocessed: 0,
    },
  },
  cuda: 0,
  data_loader: {
    additional: {
      max_decoder_source_length: 512,
      max_source_length: 512,
      max_target_length: 512,
    },
    dataset_modules: {
      module_dict: {
        LoadDataLoaders: {
          config: {
            test: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'WikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
              },
            ],
            train: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'train',
                use_column: 'wtq_data',
              },
            ],
            valid: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'WikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
              },
            ],
          },
          option: 'default',
          type: 'LoadDataLoaders',
        },
        LoadWikiSQLData: {
          config: {
            path: {
              test: 'TableQA_data/wikisql/preprocessed_test.arrow',
              train: 'TableQA_data/wikisql/preprocessed_train.arrow',
              validation: 'TableQA_data/wikisql/preprocessed_validation.arrow',
            },
            preprocess: [],
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
            preprocess: [
              'transform_to_sqa_format',
              'check_tapas_tokenization_compatibility',
            ],
          },
          option: 'default',
          type: 'LoadWikiTQData',
        },
      },
      module_list: [
        'LoadWikiTQData',
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
  ],
  model_config: {
    ConfigClass: 'TapasConfig',
    ConfigModelVersion: 'google/tapas-base-finetuned-wtq',
    ConfigPredefinedSet: 'WTQ',
    ModelClass: 'TapasForQuestionAnswering',
    ModelVersion: 'google/tapas-base-finetuned-wikisql-supervised',
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    TokenizerClass: 'CustomTapasTokenizer',
    TokenizerModelVersion: 'google/tapas-base',
    base_model: 'TAPAS',
    decoder_input_modules: {
      module_list: [],
      postprocess_module_list: [],
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
        {
          option: 'default',
          type: 'TableInput',
        },
        {
          option: 'default',
          type: 'TAPASSpecificInput',
        },
      ],
      postprocess_module_list: [
        {
          option: 'default',
          type: 'PostProcessTAPASInputTokenization',
        },
      ],
    },
    modules: [],
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
      label_smoothing_factor: 0,
      plugins: [],
      save_top_k: -1,
      save_top_k_metric: 'valid/WikiTQDataset.validation/denotation_accuracy',
      save_top_k_mode: 'max',
      warmup_steps: 0,
      weight_decay: 0,
    },
    batch_size: 32,
    epochs: 9999,
    load_best_model: 0,
    load_epoch: -1,
    load_model_path: '',
    lr: 0.0001,
    save_interval: 200,
    scheduler: 'none',
    type: 'TAPASExecutor',
  },
  valid: {
    additional: {},
    batch_size: 32,
    break_interval: 3000,
    step_size: 200,
  },
}
