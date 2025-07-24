from models import register, ParamEnc
from utils import split_str

@register('tnerv_enc')
class TNeRVEnc(ParamEnc):
    def __init__(self, tokenizer, encoder, decoder, kl_layer, vq_layer):
        self.z_token_list = split_str(decoder['z_tokens'])
        decoder['z_tokens'] = sum(self.z_token_list)
        super().__init__(tokenizer, encoder, decoder, kl_layer, vq_layer)
        self.cumsum_z = decoder['cumsum_z']
        self.sequential_decode = decoder['sequential_decode']
