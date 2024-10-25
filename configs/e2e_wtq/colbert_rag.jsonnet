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
      max_decoder_source_length: 1024,
      max_source_length: 32,
      max_target_length: 128,
      num_knowledge_passages: 5,
    },
    dataset_modules: {
      module_dict: {
        LoadDataLoaders: {
          config: {
            test: [
              {
                dataset_type: 'RAGE2EWTQDataset',
                split: 'validation',
                use_column: 'e2e_wtq_data',
              },
              {
                dataset_type: 'RAGE2EWTQDataset',
                split: 'test',
                use_column: 'e2e_wtq_data',
              },
            ],
            train: [
              {
                dataset_type: 'RAGE2EWTQDataset',
                split: 'train',
                use_column: 'e2e_wtq_data',
              },
            ],
            valid: [
              {
                dataset_type: 'RAGE2EWTQDataset',
                split: 'validation',
                use_column: 'e2e_wtq_data',
              },
              {
                dataset_type: 'RAGE2EWTQDataset',
                split: 'test',
                use_column: 'e2e_wtq_data',
              },
            ],
          },
          option: 'default',
          type: 'LoadDataLoaders',
        },
        LoadE2EWTQData: {
          config: {
            bm25_results: 'TableQA_data/e2e_wtq/e2e_wtq_bm25_results.json',
            data_path: {
              test: 'TableQA_data/e2e_wtq/test_lookup.jsonl.gz',
              train: 'TableQA_data/e2e_wtq/train_lookup.jsonl.gz',
              validation: 'TableQA_data/e2e_wtq/dev_lookup.jsonl.gz',
            },
            path: {
              tables: 'TableQA_data/e2e_wtq/preprocessed_tables.arrow',
              test: 'TableQA_data/e2e_wtq/preprocessed_test.arrow',
              train: 'TableQA_data/e2e_wtq/preprocessed_train.arrow',
              validation: 'TableQA_data/e2e_wtq/preprocessed_validation.arrow',
            },
            preprocess: [],
          },
          option: 'default',
          type: 'LoadE2EWTQData',
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
            preprocess: [],
          },
          option: 'default',
          type: 'LoadWikiTQData',
        },
      },
      module_list: [
        'LoadE2EWTQData',
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
      name: 'compute_denotation_accuracy',
      squad_normalization: 0,
      vague: 1,
    },
    {
      name: 'compute_RAG_retrieval_results',
    },
    {
      name: 'compute_token_f1',
    },
  ],
  model_config: {
    DECODER_SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    DecoderTokenizerClass: 'TapexTokenizer',
    DecoderTokenizerModelVersion: 'microsoft/tapex-large',
    GeneratorConfigClass: 'BartConfig',
    GeneratorModelClass: 'BartForConditionalGeneration',
    GeneratorModelVersion: 'microsoft/tapex-large',
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
    ModelClass: 'RagModel',
    QueryEncoderModelClass: 'ColBERT',
    QueryEncoderModelVersion: '/wd/Experiments/ColBERT_E2EWTQ_bz4_negative4_fix_doclen_full_search_NewcrossGPU/train/saved_model/step_300',
    RAVQA_loss_type: 'Approach5',
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    TokenizerClass: 'QueryTokenizer',
    base_model: 'RAG',
    decoder_input_modules: {
      module_list: [],
      postprocess_module_list: [],
    },
    index_files: {
      embedding_path: 'ColBERT_E2EWTQ_bz4_negative4_fix_doclen_full_search_NewcrossGPU/test/e2e_wtq_all/step_300/item_embeddings.pkl',
      index_passages_path: 'ColBERT_E2EWTQ_bz4_negative4_fix_doclen_full_search_NewcrossGPU/test/e2e_wtq_all/step_300/table_dataset',
      index_path: 'ColBERT_E2EWTQ_bz4_negative4_fix_doclen_full_search_NewcrossGPU/test/e2e_wtq_all/step_300/table_dataset_colbert_index',
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
          type: 'PostProcessColBERTQuestionInputTokenization',
        },
      ],
    },
    loss_ratio: {
      additional_loss: 1,
      nll_loss: 1,
      rag_loss: 0,
    },
    modules: [],
    num_beams: 5,
    output_modules: {
      module_list: [
        {
          option: 'default',
          type: 'FlattenedAnswerOutput',
        },
      ],
      postprocess_module_list: [
        {
          option: 'default',
          type: 'PostProcessTAPEXOutputTokenization',
        },
      ],
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
      save_top_k_metric: 'valid/RAGE2EWTQDataset.validation/denotation_accuracy',
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
    retriever_lr: 1.0000000000000001e-05,
    save_interval: 1000,
    scheduler: 'none',
    type: 'RAGExecutor',
  },
  valid: {
    additional: {},
    batch_size: 32,
    break_interval: 3000,
    step_size: 1000,
  },
}
