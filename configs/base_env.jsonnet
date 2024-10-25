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
    additional: {},
    dataset_modules: {
      module_dict: {
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
            preprocess: [],
          },
          option: 'default',
          type: 'LoadWikiTQData',
        },
      },
      module_list: [],
    },
    datasets: {},
    dummy_dataloader: 0,
    type: 'DataLoaderForTableQA',
  },
  experiment_name: 'default_test',
  gpu_device: 0,
  ignore_pretrained_weights: [],
  model_config: {
    base_model: 'TAPAS',
    decoder_input_modules: {
      module_list: [],
      postprocess_module_list: [],
    },
    input_modules: {
      module_list: [],
      postprocess_module_list: [],
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
  seed: 2021,
  test: {
    additional: {
      multiprocessing: 4,
    },
    batch_size: 32,
    evaluation_name: 'test_evaluation',
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
      plugins: [],
      save_top_k: -1,
      save_top_k_metric: 'valid/denotation_accuracy',
      save_top_k_mode: 'max',
      warmup_steps: 0,
    },
    batch_size: 32,
    epochs: 9999,
    load_epoch: -1,
    load_model_path: '',
    lr: 0.0001,
    save_interval: 1,
    scheduler: 'none',
    type: 'TAPASExecutor',
  },
  valid: {
    additional: {},
    batch_size: 32,
    step_size: 100,
  },
}
