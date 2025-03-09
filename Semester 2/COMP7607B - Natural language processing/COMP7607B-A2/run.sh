pytest test/test_attention.py
pytest test/test_dpo.py

python train_pretrain.py
python train_sft.py
python train_lora.py
python train_dpo.py