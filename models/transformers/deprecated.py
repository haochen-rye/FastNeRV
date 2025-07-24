
# Deprecated

# Vid2NeRV_DiT
class Vid2NeRV_DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_dropout_prob=0.1,
        num_classes=1000,
        proj_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, cond_dropout_prob)
        self.hid_dim = hidden_size

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        return x


def Vid2NeRV_DiT_H(**kwargs):
    return Vid2NeRV_DiT(depth=32, hidden_size=1280, num_heads=16, **kwargs)

def Vid2NeRV_DiT_XL(**kwargs):
    return Vid2NeRV_DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def Vid2NeRV_DiT_L(**kwargs):
    return Vid2NeRV_DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def Vid2NeRV_DiT_B(**kwargs):
    return Vid2NeRV_DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def Vid2NeRV_DiT_S(**kwargs):
    return Vid2NeRV_DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

Vid2NeRV_DiT_models = {'ViT-H': Vid2NeRV_DiT_H,
    'ViT-XL': Vid2NeRV_DiT_XL, 'ViT-L':  Vid2NeRV_DiT_L,  
    'ViT-B':  Vid2NeRV_DiT_B,  'ViT-S':  Vid2NeRV_DiT_S,  
}
