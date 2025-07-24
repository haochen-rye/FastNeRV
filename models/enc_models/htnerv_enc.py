from models import register, TNeRVEnc
from utils import split_str

@register('htnerv_enc')
class HTNeRVEnc(TNeRVEnc):
    def __init__(self, tokenizer, encoder, decoder, kl_layer, vq_layer):
        super().__init__(tokenizer, encoder, decoder, kl_layer, vq_layer)
        self.cumsum_z = decoder['cumsum_z']
        self.sequential_decode = decoder['sequential_decode']
     
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
