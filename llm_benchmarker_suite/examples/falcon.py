from mmengine.config import read_base

with read_base():
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

datasets = [*siqa_datasets, *winograd_datasets]

from opencompass.models import HuggingFaceCausalLM

# Falcon-40B
falcon = dict(
       type=HuggingFaceCausalLM,
       # the folowing are HuggingFaceCausalLM init parameters
       path='tiiuae/falcon-40b',
       tokenizer_path='tiiuae/falcon-40b',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       max_seq_len=2048,
       # the folowing are not HuggingFaceCausalLM init parameters
       abbr='tiiuae/falcon-40b',                # Model abbreviation
       max_out_len=100,               # Maximum number of generated tokens
       batch_size=128,
       run_cfg=dict(num_gpus=1),   # Run configuration for specifying resource requirements
    )

models = [falcon]