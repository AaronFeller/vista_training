from transformers import PretrainedConfig

class model_config(PretrainedConfig):
    model_type = "MLM_model"
    def __init__(
        self,
        ffn_hidden_dim = 1024,
        embed_dim = 768,
        num_heads = 12,
        num_blocks = 24,
        vocab_size = 405,
        output_dim = 405,
        max_seq_len = 2048,
        size = "base",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ffn_hidden_dim = ffn_hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.size = size