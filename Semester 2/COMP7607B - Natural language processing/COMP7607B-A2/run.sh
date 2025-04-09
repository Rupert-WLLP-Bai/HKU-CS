pytest test/test_attention.py
pytest test/test_dpo.py

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' python train_pretrain.py
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' python train_sft.py
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' python train_lora.py
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' python train_dpo.py