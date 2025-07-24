import torch
from models import register, NeRVEnc
from models.transformers.vit_model import DiffuseViT_models
from utils import DiagonalGaussianDistribution

@register('diffuse_v2n')
class DiffuseV2N(NeRVEnc):
    def __init__(self, tokenizer, hyponet, nerv_tokens, transformer_encoder, kl_layer):
        frame_num, vid_size, patch_size = [tokenizer['args'][x] for x in ['frame_num', 'input_size', 'patch_hw']]
        transformer_encoder['num_patch'] = frame_num * (vid_size // patch_size)**2
        transformer_encoder['in_channels'] = hyponet['args']['out_dim']
        self.cond_frames = transformer_encoder['cond_frames']
        if self.cond_frames != 'no':
            tokenizer['args']['in_ch'] = 6
        # tokenizer['args']['frame_num'] = tokenizer['args']['frame_num'] + self.cond_frames
        super().__init__(tokenizer, hyponet, nerv_tokens, transformer_encoder, kl_layer)
        self.z_num = frame_num * vid_size**2
        self.vid_t, self.vid_h, self.vid_w = frame_num, vid_size, vid_size


    def create_transformer(self, transformer_encoder):
        model = DiffuseViT_models[transformer_encoder['name']](
            cond_dropout_prob = transformer_encoder['cond_dropout_prob'],
            num_patch = transformer_encoder['num_patch'],
            sum_t_y = transformer_encoder['sum_t_y'],
            att_every_block = transformer_encoder['att_every_block'],
            in_channels = transformer_encoder['in_channels'],
            learn_sigma = transformer_encoder['learn_sigma'],
            final_layer = transformer_encoder['final_layer'],
            final_modulate = transformer_encoder['final_modulate'],
            final_bias = transformer_encoder['final_bias'],
            out_range = transformer_encoder['out_range'],
        )      
        return model  


    def encode(self, data, t, y, c):
        # Video tokenization and go through transformer encoder
        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        if c != None:
            data = torch.cat([c, data], dim=1)
        # dtokens = self.tokenizer([c, data] if c!=None else data)
        dtokens = self.tokenizer(data)
        assert self.z_type in ['nerv', 'no']
        wtokens = self.repeat_input_tokens(self.wtokens, dtokens.size(0))
        trans_out = self.transformer_encoder([dtokens, wtokens], t=t, y=y)
        z_tokens = self.quant_z_fc(trans_out[:, -len(self.wtokens):])

        # if self.z_type in ['no']:
        if True:
            z, kl_loss = z_tokens, 0
        else:
            NotImplementedError
            z_tokens = self.z_norm_layer(z_tokens)   # Normalize the bottleneck_z
            z_posterior = DiagonalGaussianDistribution(z_tokens, self.deterministic)
            z = z_posterior.sample()     
            kl_loss = self.get_kl_loss(z_posterior)
        
        return {'z': z, 'kl_loss': kl_loss}


    def forward(self, x, t, y, c):
        z_info = self.encode(x, t, y, c)
        x_hat = self.decode(z_info['z'])['pred_frames']
        x_hat = self.transformer_encoder.final_forward(x_hat, t, y)
        return x_hat
