"""Published model-config presets used by the layer planner.

The values below are copied from the public Hugging Face ``config.json`` files
for the model families. We intentionally keep only fields consumed by the
planner; omitted fields are not silently inferred as published facts.
"""

LLAMA_3_8B = {
    "_name_or_path": "meta-llama/Meta-Llama-3-8B",
    "model_type": "llama",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "source": "https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json",
}

LLAMA_3_70B = {
    "_name_or_path": "meta-llama/Meta-Llama-3-70B",
    "model_type": "llama",
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "source": "https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json",
}

MIXTRAL_8X7B = {
    "_name_or_path": "mistralai/Mixtral-8x7B-v0.1",
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "source": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json",
}

BUILTIN_MODEL_CONFIGS = {
    "llama3_8b": LLAMA_3_8B,
    "llama3_70b": LLAMA_3_70B,
    "mixtral_8x7b": MIXTRAL_8X7B,
}
