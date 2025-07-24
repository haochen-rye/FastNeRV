import torch


def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb.to(x))

def batched_conv(x, wb, conv_shape, ps_layer):
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad = conv_shape
    conv_weigts = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weigts = conv_weigts.permute(0,-1,1,2,3).flatten(end_dim=1)
    # import pdb; pdb.set_trace; from IPython import embed; embed()      
    conv_bias = wb[:,-1:].flatten()
    x = torch.nn.functional.conv2d(x, conv_weigts, conv_bias, stride=1, padding=pad, groups=B)
    return ps_layer(x)