import einops
import torch.nn as nn

from models import register
from utils.common import get_sincos_pos_embed

@register('vid_tokenizer')
class VidTokenizer(nn.Module):
    def __init__(self, input_size, patch_hw, patch_t, dim, 
                 in_ch=3, frame_num=16, learn_pe=False):
        super().__init__()
        nt, nh, nw = frame_num//patch_t, input_size//patch_hw, input_size//patch_hw
        self.num_t, self.num_h, self.num_w = nt, nh, nw
        self.patch_num = nt*nh*nw
        self.frame_num, self.input_size, self.patch_t, self.patch_hw = frame_num, input_size, patch_t, patch_hw
        self.prefc = nn.Conv3d(in_ch, dim, kernel_size=(patch_t, patch_hw, patch_hw), 
            stride=(patch_t, patch_hw, patch_hw), )
        pos_embed = get_sincos_pos_embed(dim, (nt, nh, nw)).view(1, self.patch_num, dim)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=learn_pe)
        self.initialize_weights()


    def initialize_weights(self):    
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.prefc.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.prefc.bias, 0)


    def forward(self, data):
        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        x = self.prefc(data) 
        x = einops.rearrange(x, 'b c t h w->b (t h w) c') + self.pos_embed
        return x
    