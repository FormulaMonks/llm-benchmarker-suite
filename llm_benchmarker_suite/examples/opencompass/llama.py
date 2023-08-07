from mmengine.config import read_base

with read_base():
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

datasets = [*siqa_datasets, *winograd_datasets]

from opencompass.models import HuggingFaceCausalLM

# Llama
llama = dict(
       type=HuggingFaceCausalLM,
       # the following are HuggingFaceCausalLM init parameters
       path='meta-llama/Llama-2-70b-hf',
       tokenizer_path='meta-llama/Llama-2-70b-hf',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       max_seq_len=2048,
       # the following are not HuggingFaceCausalLM init parameters
       abbr='meta-llama/Llama-2-70b-hf',                # Model abbreviation
       max_out_len=100,               # Maximum number of generated tokens
       batch_size=128,
       run_cfg=dict(num_gpus=1),   # Run configuration for specifying resource requirements
    )

models = [llama]
