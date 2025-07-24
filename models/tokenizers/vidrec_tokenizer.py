import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding3D, Summer

from models import register


@register('vidrec_tokenizer')
class VidrecTokenizer(nn.Module):
    def __init__(self, input_size, patch_size, dim, frame_num=16,
             padding=0, img_channels=3, img_groups='1_1', pe='learned'):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.input_size, self.patch_size, self.padding = input_size, patch_size, padding
        self.n_frames = frame_num
        clip_len, patch_t = [int(x) for x in img_groups.split('_')[:2]]
        self.len_t, self.len_h, self.len_w = self.n_frames//clip_len, input_size[0]//patch_size[0], input_size[1]//patch_size[1]
        self.patch_num = self.len_t * self.len_h * self.len_w
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels * patch_t, dim)
        if pe=='learned':
            self.posemb = nn.Parameter(torch.randn(self.patch_num, dim))
        elif pe == 'fixed':
            self.posemb = Summer(PositionalEncoding3D(dim))
        else:
            raise ValueError(f'Unsupported position embedding {pe} for input tokens')
        self.clip_len, self.patch_t = clip_len, patch_t
        self.pe = pe
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def get_post_fc(self):
        in_f, out_f = self.prefc.in_features, self.prefc.out_features 
        return nn.Sequential(
            nn.LayerNorm(out_f),
            nn.Linear(out_f, in_f),            
        )
    
    
    def forward(self, data):
        if isinstance(data, list):
            data = torch.cat(data, dim=2)      # concat x_t and cond_frames in t dimension
        x_tokens = self.tokenize_video(data)
        if self.pe == 'learned':
            x = self.prefc(x_tokens) + self.posemb.unsqueeze(0)
        else:
            x = self.posemb(self.prefc(x_tokens))
        return x
    
    def tokenize_video(self, data):
        x_tokens = data.view(data.size(0), 3, self.len_t, self.patch_t, self.len_h, self.patch_size[0], self.len_w, self.patch_size[1])
        x_tokens = einops.rearrange(x_tokens, 'b c t pt h ph w pw -> b (t h w) (c pt ph pw)')   # (B, T*H*W, C * t * p * p)
        return x_tokens


    def token2vid(self, tokens, time_first=True):
        token_shape = (tokens.size(0), self.len_t, self.len_h, self.len_w, 3, self.patch_t) + self.patch_size
        vid_shape = (tokens.size(0), self.n_frames, 3) + self.input_size
        x = einops.rearrange(tokens.view(token_shape), 'b t h w c tn hn wn -> b t tn c h hn w wn')
        x = x.reshape(vid_shape)
        if time_first:
            x = einops.rearrange(x, 'b t c h w -> t b c h w')
        # import pdb; pdb.set_trace; from IPython import embed; embed()   

        return x




# Deprecated
@register('old_vidrec_tokenizer')
class OldVidrecTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, frame_num=16,
             padding=0, img_channels=3, img_groups='1_1', pe='learned'):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.input_size = input_size
        self.patch_size = patch_size
        self.n_frames = frame_num
        self.padding = padding
        clip_len, patch_t = [int(x) for x in img_groups.split('_')[:2]]
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels * patch_t, dim)
        # clip_num = frame_num // img_groups
        if pe=='learned':
            n_patches = 1
            for dim_id in range(2):
                n_patches *= (input_size[dim_id] + self.padding[dim_id] * 2) // self.patch_size[dim_id]
            self.posemb = nn.Parameter(torch.randn(frame_num // clip_len, n_patches, dim))
        elif pe == 'fixed':
            self.posemb = Summer(PositionalEncoding3D(dim))
        else:
            raise ValueError(f'Unsupported position embedding {pe} for input tokens')
        self.img_groups, self.clip_len, self.patch_t = img_groups, clip_len, patch_t
        self.pe = pe

    def forward(self, data):
        p = self.patch_size
        x_batch, c, t, h, w = data.size()
        clip_idx = torch.arange(self.patch_t)
        if self.pe == 'learned':
            x = data.view(x_batch, c, t//self.clip_len, self.clip_len, h, w)[:,:,:,clip_idx].permute(0,2,1,3,-2,-1)
            x = F.unfold(x.flatten(end_dim=1).flatten(start_dim=1,end_dim=2), p, stride=p, padding=self.padding) # (B*T, C * patch_t * p * p, L)
            x = x.view((x_batch, -1)+x.shape[1:]).permute(0, 1, 3, 2).flatten(start_dim=1, end_dim=2).contiguous()
            x = self.prefc(x) + self.posemb.flatten(end_dim=1).unsqueeze(0)
        else:
            # import pdb; pdb.set_trace; from IPython import embed; embed()   
            x_tokens = data.view(x_batch, c, t//self.clip_len, self.clip_len, h//p[0], p[0], w//p[1], p[1])[:,:,:,clip_idx]
            x_tokens = x_tokens.permute(0,2,4,6,1,3,5,7).flatten(start_dim=-4) # (B, T, H, W, C * t * p * p)
            x = self.posemb(self.prefc(x_tokens)).flatten(start_dim=1,end_dim=3)
        return x
    
    def token2vid(self, tokens):
        token_shape = (tokens.size(0), self.n_frames//self.patch_t, self.input_size[0]//self.patch_size[0], self.input_size[1]//self.patch_size[1], 3, self.patch_t) + self.patch_size
        vid_shape = (tokens.size(0), self.n_frames, 3) + self.input_size
        x = tokens.view(token_shape).permute(0,1,-3,4,2,-2,3,-1).reshape(vid_shape)
        return x
