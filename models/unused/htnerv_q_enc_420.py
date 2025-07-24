from models import register, HTNeRVEnc
from utils import VQEmbedding
import torch.nn.functional as F

@register('htnerv_q_enc')
class HTNeRVQEnc(HTNeRVEnc):
    def __init__(self, tokenizer, encoder, decoder, kl_layer, vq_layer):
        super().__init__(tokenizer, encoder, decoder, kl_layer)  
        self.use_vq = vq_layer['use_vq']   
        if self.use_vq:
            self.z_codebook = VQEmbedding(vq_layer['codebook_size'],
                                        self.z_dim, 
                                        using_znorm = vq_layer['using_znorm'],
                                        unuse_token_thres = vq_layer['unuse_token_thres'],
                                        ema = vq_layer['ema'],
                                        decay = vq_layer['ema_decay'], 
                                        )
            self.vq_weight = vq_layer['vq_weight']
            self.commit_loss = vq_layer['commit_loss']
            self.codebook_loss = vq_layer['codebook_loss']
        # import pdb; pdb.set_trace; from IPython import embed; embed()  

    def bottle_forward(self, z):
        z_info = super().bottle_forward(z)
        if self.use_vq:
            z_orig = z_info.pop('z')
            z_quant, z_code = self.z_codebook(z_orig)
            commit_loss = F.mse_loss(z_quant.detach(), z_orig) 
            codebook_loss = F.mse_loss(z_quant, z_orig.detach())
            vq_loss = self.vq_weight * (self.commit_loss * commit_loss + self.codebook_loss * codebook_loss)
            # * self.codebook_loss + \
            #     F.mse_loss(z_quant.detach(), z_orig) * self.commit_loss
            z_hat = (z_quant - z_orig).detach() + z_orig
            z_info.update({
                'z': z_hat, 'z_code': z_code,
                'commit_loss': commit_loss, 'codebook_loss': codebook_loss, 'vq_loss': vq_loss, 
            })
        return z_info 
        # return {**z_info, 'base_vq': commit_loss, 'vq_loss': self.vq_weight*commit_loss, 
        #         'z': z_hat, 'z_code': z_code}

