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
    },
    dataset_modules: {
      module_dict: {
        LoadDataLoaders: {
          config: {
            test: [
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'train',
                use_column: 'wikisql_data',
              },
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'validation',
                use_column: 'wikisql_data',
              },
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'test',
                use_column: 'wikisql_data',
              },
            ],
            train: [
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'train',
                use_column: 'wikisql_data',
              },
            ],
            valid: [
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'validation',
                use_column: 'wikisql_data',
              },
              {
                dataset_type: 'ITRWikiSQLDataset',
                split: 'test',
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
              test: 'TableQA_data/wikisql/preprocessed_move_answers_to_table_end_split_table_by_mixed_combination_test.arrow',
              train: 'TableQA_data/wikisql/preprocessed_move_answers_to_table_end_split_table_by_mixed_combination_train.arrow',
              validation: 'TableQA_data/wikisql/preprocessed_move_answers_to_table_end_split_table_by_mixed_combination_validation.arrow',
            },
            preprocess: [
              'move_answers_to_table_end',
              'split_table_by_mixed_combination',
            ],
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
      name: 'compute_ITR_mix_retrieval_results',
      option: 'mix',
    },
  ],
  model_config: {
    DECODER_SPECIAL_TOKENS: {
      additional_special_tokens: [
        '<HEADER>',
        '<HEADER_SEP>',
        '<HEADER_END>',
        '<ROW>',
        '<ROW_SEP>',
        '<ROW_END>',
      ],
    },
    DecoderTokenizerClass: 'DPRContextEncoderTokenizer',
    DecoderTokenizerModelVersion: 'facebook/dpr-ctx_encoder-single-nq-base',
    ItemEncoderConfigClass: 'DPRConfig',
    ItemEncoderModelClass: 'DPRContextEncoder',
    ItemEncoderModelVersion: 'facebook/dpr-ctx_encoder-single-nq-base',
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
    ModelClass: 'RetrieverDPR',
    QueryEncoderConfigClass: 'DPRConfig',
    QueryEncoderModelClass: 'DPRQuestionEncoder',
    QueryEncoderModelVersion: 'facebook/dpr-question_encoder-single-nq-base',
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    TokenizerClass: 'DPRQuestionEncoderTokenizer',
    TokenizerModelVersion: 'facebook/dpr-question_encoder-single-nq-base',
    base_model: 'DPR',
    decoder_input_modules: {
      module_list: [
        {
          option: 'default',
          separation_tokens: {
            header_end: '<HEADER_END>',
            header_sep: '<HEADER_SEP>',
            header_start: '<HEADER>',
            row_end: '<ROW_END>',
            row_sep: '<ROW_SEP>',
            row_start: '<ROW>',
          },
          type: 'TextBasedTableInput',
        },
      ],
      postprocess_module_list: [
        {
          option: 'default',
          type: 'PostProcessDecoderInputTokenization',
        },
      ],
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
    modules: [
      'separate_query_and_item_encoders',
    ],
    num_negative_samples: 4,
    output_modules: {
      module_list: [
        {
          option: 'default',
          type: 'SimilarityOutput',
        },
      ],
      postprocess_module_list: [
        {
          option: 'default',
          type: 'PostProcessConcatenateLabels',
        },
      ],
    },
    prepend_tokens: {
      item_encoder: '',
      query_encoder: '',
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
      plugins: [],
      save_top_k: -1,
      save_top_k_metric: 'valid/ITRWikiSQLDataset.validation/full_recall_at_5',
      save_top_k_mode: 'max',
      warmup_steps: 0,
    },
    batch_size: 32,
    epochs: 9999,
    load_best_model: 0,
    load_epoch: -1,
    load_model_path: '',
    lr: 0.0001,
    save_interval: 200,
    scheduler: 'none',
    type: 'ITRDPRExecutor',
  },
  valid: {
    additional: {},
    batch_size: 32,
    break_interval: 3000,
    step_size: 200,
  },
}
