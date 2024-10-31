""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
from transformers import AutoConfig
from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LlamaBitNetConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
    ExistingCheckpointInit
)
from nanotron.logging import human_format

# model_config = LlamaConfig(
#     # Config for a tiny model model with 1.62M parameters
#     bos_token_id=1,
#     eos_token_id=2,
#     hidden_act="silu",
#     hidden_size=16,
#     initializer_range=0.02,
#     intermediate_size=64,
#     max_position_embeddings=256,
#     num_attention_heads=4,
#     num_hidden_layers=2,
#     num_key_value_heads=4,
#     pretraining_tp=1,
#     rms_norm_eps=1e-05,
#     rope_scaling=None,
#     tie_word_embeddings=True,
#     use_cache=True,
#     vocab_size=256,
# )
hf_config = AutoConfig.from_pretrained('/fl-ift/med/common/Qwen2.5-72B-Instruct')
# model_config = LlamaBitNetConfig(
#     # Config for a tiny 1.58bit model model with 1.62M parameters
#     bos_token_id=1,
#     eos_token_id=2,
#     hidden_act="silu",
#     hidden_size=16,
#     initializer_range=0.02,
#     intermediate_size=64,
#     is_bitnet_config=True,
#     max_position_embeddings=256,
#     num_attention_heads=4,
#     num_hidden_layers=2,
#     num_key_value_heads=4,
#     pretraining_tp=1,
#     rms_norm_eps=1e-05,
#     rope_scaling=None,
#     tie_word_embeddings=True,
#     use_cache=True,
#     vocab_size=256,
# )
model_config = LlamaBitNetConfig(
    # bos_token_id=hf_config.bos_token_id,
    # eos_token_id=hf_config.eos_token_id,
    hidden_act=hf_config.hidden_act,
    hidden_size=hf_config.hidden_size,
    initializer_range=hf_config.initializer_range,
    intermediate_size=hf_config.intermediate_size,
    is_bitnet_config=True,
    max_position_embeddings=hf_config.max_position_embeddings,
    num_attention_heads=hf_config.num_attention_heads,
    num_hidden_layers=hf_config.num_hidden_layers,
    num_key_value_heads=hf_config.num_key_value_heads,
    pretraining_tp=1,
    rms_norm_eps=hf_config.rms_norm_eps,
    rope_scaling=hf_config.rope_scaling,
    tie_word_embeddings=hf_config.tie_word_embeddings,
    use_cache=hf_config.use_cache,
    vocab_size=hf_config.vocab_size,
)

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        3 * model_config.hidden_size * model_config.intermediate_size
        + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")

seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=2e-5, lr_warmup_steps=20, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-6
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=1,
    pp=8,
    tp=1,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
)

tokens = TokensArgs(sequence_length=4096, train_steps=443, micro_batch_size=4, batch_accumulation_per_replica=4)

data_stages = [
    DatasetStageArgs(
        name="Stable Training Stage",
        start_training_step=1,
        data=DataArgs(
            dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="data/radiology", text_column_name="text"),
            seed=seed,
        ),
    ),
    DatasetStageArgs(
        name="Annealing Phase",
        start_training_step=10,
        data=DataArgs(
            dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="data/radiology", text_column_name="text"),
            seed=seed,
        ),
    ),
]

checkpoints_path = "./checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="debug", run="tiny_llama_%date_%jobid", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=100),
    parallelism=parallelism,
    model=ModelArgs(init_method=ExistingCheckpointInit(path='/fl-ift/med/hujunchao/models/qwen2.5-72b-instruct-nanotron/'),model_config=model_config),
    tokenizer=TokenizerArgs("/fl-ift/med/hujunchao/models/qwen2.5-72b-instruct-nanotron/"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=data_stages,
    profiler=None,
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_bitnet_qw2.yaml")

    # You can now train a model with this config using `/run_train.py`
