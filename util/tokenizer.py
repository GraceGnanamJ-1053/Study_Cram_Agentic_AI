import os

from transformers import AutoTokenizer

# if "wandb_mode" not in os.environ:
#     local_files_only = True
# else:
#     local_files_only = os.environ["WANDB_MODE"] == "offline"

# TOKENIZERS = {
#     "smollm": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=local_files_only),
#     "smollm2": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=local_files_only),
# }

SHARED_CHECKPOINT_PATH = "checkpoints/250720_pretrain_smollm-360m_kv-share_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001"
TOKENIZERS = {
    # CHANGE 1: Use your local folder path instead of the Hugging Face ID
    "smollm": AutoTokenizer.from_pretrained(SHARED_CHECKPOINT_PATH, local_files_only=True), 
    
    # CHANGE 2: Add the needed 360M model using its local folder path
    "smollm-360m": AutoTokenizer.from_pretrained(SHARED_CHECKPOINT_PATH, local_files_only=True), 
    
    # CHANGE 3: Update other tokens, pointing to its local path
    "smollm2": AutoTokenizer.from_pretrained(SHARED_CHECKPOINT_PATH, local_files_only=True),
}

def load_tokenizer_from_config(cfg):
    tokenizer = TOKENIZERS[cfg.tokenizer]
    if tokenizer.pad_token is None:
        if cfg.tokenizer in ["smollm", "smollm2"]:
            # '<|endoftext|>'
            tokenizer.pad_token_id = 0
        else:
            raise ValueError(f"Tokenizer {cfg.tokenizer} does not have a pad token, please specify one in the config")
    return tokenizer

