from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from constants import MODEL_CHECKPOINT


def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for token classification.

    Returns:
        A tokenizer for token classification.
    NOTE: Below is an example of how to initialize a tokenizer for token classification. You are free to change this.
    # But make sure the tokenizer is the same as the model.
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    return tokenizer
