import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

from .uformer import Uformer, Downsample, Upsample, InputProj, OutputProj, BasicUformerLayer


class sstfusion(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False,
                 cross_modulator=False, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x1, x2, mask=None):
        # Input Projection
        fusion_rule = l1fusion()

        x1 = self.input_proj(x1)
        x1 = self.pos_drop(x1)

        x2 = self.input_proj(x2)
        x2 = self.pos_drop(x2)
        # Encoder
        x1_conv0 = self.encoderlayer_0(x1, mask=mask)  # 1 65536 16
        x1_pool0 = self.dowsample_0(x1_conv0)  # 1 16384 32
        x1_conv1 = self.encoderlayer_1(x1_pool0, mask=mask)  # 1 16384 32
        x1_pool1 = self.dowsample_1(x1_conv1)  # 1 4096 64
        x1_conv2 = self.encoderlayer_2(x1_pool1, mask=mask)  # 1 4096 64
        x1_pool2 = self.dowsample_2(x1_conv2)  # 1 1024 128
        x1_conv3 = self.encoderlayer_3(x1_pool2, mask=mask)  # 1 1024 128
        x1_pool3 = self.dowsample_3(x1_conv3)  # 1 256 256

        x2_conv0 = self.encoderlayer_0(x2, mask=mask)  # 1 65536 16
        x2_pool0 = self.dowsample_0(x2_conv0)  # 1 16384 32
        x2_conv1 = self.encoderlayer_1(x2_pool0, mask=mask)  # 1 16384 32
        x2_pool1 = self.dowsample_1(x2_conv1)  # 1 4096 64
        x2_conv2 = self.encoderlayer_2(x2_pool1, mask=mask)  # 1 4096 64
        x2_pool2 = self.dowsample_2(x2_conv2)  # 1 1024 128
        x2_conv3 = self.encoderlayer_3(x2_pool2, mask=mask)  # 1 1024 128
        x2_pool3 = self.dowsample_3(x2_conv3)  # 1 256 256

        # Bottleneck
        x1_conv4 = self.conv(x1_pool3, mask=mask)  # 1 256 256

        x2_conv4 = self.conv(x2_pool3, mask=mask)  # 1 256 256

        # Fusion Rule
        x1_conv4 = rearrange(x1_conv4, 'b (h w) c -> b c h w', h=int(math.sqrt(x1_conv4.shape[1])),
                             w=int(math.sqrt(x1_conv4.shape[1])))
        x2_conv4 = rearrange(x2_conv4, 'b (h w) c -> b c h w', h=int(math.sqrt(x2_conv4.shape[1])),
                             w=int(math.sqrt(x2_conv4.shape[1])))
        conv4 = fusion_rule(x1_conv4, x2_conv4)
        conv4 = rearrange(conv4, 'b c h w -> b (h w) c')
        x1_conv3 = rearrange(x1_conv3, 'b (h w) c -> b c h w', h=int(math.sqrt(x1_conv3.shape[1])),
                             w=int(math.sqrt(x1_conv3.shape[1])))
        x2_conv3 = rearrange(x2_conv3, 'b (h w) c -> b c h w', h=int(math.sqrt(x2_conv3.shape[1])),
                             w=int(math.sqrt(x2_conv3.shape[1])))
        conv3 = fusion_rule(x1_conv3, x2_conv3)
        conv3 = rearrange(conv3, 'b c h w -> b (h w) c')
        x1_conv2 = rearrange(x1_conv2, 'b (h w) c -> b c h w', h=int(math.sqrt(x1_conv2.shape[1])),
                             w=int(math.sqrt(x1_conv2.shape[1])))
        x2_conv2 = rearrange(x2_conv2, 'b (h w) c -> b c h w', h=int(math.sqrt(x2_conv2.shape[1])),
                             w=int(math.sqrt(x2_conv2.shape[1])))
        conv2 = fusion_rule(x1_conv2, x2_conv2)
        conv2 = rearrange(conv2, 'b c h w -> b (h w) c')
        x1_conv1 = rearrange(x1_conv1, 'b (h w) c -> b c h w', h=int(math.sqrt(x1_conv1.shape[1])),
                             w=int(math.sqrt(x1_conv1.shape[1])))
        x2_conv1 = rearrange(x2_conv1, 'b (h w) c -> b c h w', h=int(math.sqrt(x2_conv1.shape[1])),
                             w=int(math.sqrt(x2_conv1.shape[1])))
        conv1 = fusion_rule(x1_conv1, x2_conv1)
        conv1 = rearrange(conv1, 'b c h w -> b (h w) c')
        x1_conv0 = rearrange(x1_conv0, 'b (h w) c -> b c h w', h=int(math.sqrt(x1_conv0.shape[1])),
                             w=int(math.sqrt(x1_conv0.shape[1])))
        x2_conv0 = rearrange(x2_conv0, 'b (h w) c -> b c h w', h=int(math.sqrt(x2_conv0.shape[1])),
                             w=int(math.sqrt(x2_conv0.shape[1])))
        conv0 = fusion_rule(x1_conv0, x2_conv0)
        conv0 = rearrange(conv0, 'b c h w -> b (h w) c')

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)  # 1 65536 16
        deconv3 = torch.cat([up3, conv0], -1)  # 1 65536 32
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)  # 1 1 256 256
        return y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


class l1fusion(nn.Module):
    def __init__(self, window_width=1):
        super(l1fusion, self).__init__()
        self.window_width = window_width

    def forward(self, y1, y2):
        device = y1.device
        ActivityMap1 = y1.abs()
        ActivityMap2 = y2.abs()

        kernel = torch.ones(2 * self.window_width + 1, 2 * self.window_width + 1) / (2 * self.window_width + 1) ** 2
        kernel = kernel.to(device).type(torch.float32)[None, None, :, :]
        kernel = kernel.expand(y1.shape[1], y1.shape[1], 2 * self.window_width + 1, 2 * self.window_width + 1)
        ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=self.window_width)
        ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=self.window_width)
        WeightMap1 = ActivityMap1 / (ActivityMap1 + ActivityMap2)
        WeightMap2 = ActivityMap2 / (ActivityMap1 + ActivityMap2)
        return WeightMap1 * y1 + WeightMap2 * y2


if __name__ == "__main__":
    input_size = 256
    arch = Uformer
    device = 'cuda:0'
    depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = SSTFusion(in_chans=1, dd_in=1).to(device)
    # print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print('# model_restoration parameters: %.2f M' % (
    #         sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    # print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))
    x = torch.randn((1, 1, 256, 256)).to(device)
    y = torch.randn((1, 1, 256, 256)).to(device)

    result = model_restoration(x, y)
    print(result.shape)
