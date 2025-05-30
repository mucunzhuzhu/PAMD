from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from dataset.quaternion import ax_from_6v

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = True,
            device=None,
            dtype=None,
            rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(
    nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=True,
            device=None,
            dtype=None,
            rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
            self,
            tgt,
            memory,
            t,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class FiLMTransformerDecoderLayer2(
    nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=True,
            device=None,
            dtype=None,
            rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
            self,
            tgt,
            memory,
            t,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            # x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            # x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # # feedforward -> film -> residual
            # x_3 = self._ff_block(self.norm3(x))
            # x = x + featurewise_affine(x_3, self.film3(t))
        else:
            # x = self.norm1(
            #     x
            #     + featurewise_affine(
            #         self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
            #     )
            # )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            # x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
            self,
            smpl,
            nfeats: int,
            seq_len: int = 150,
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            cond_feature_dim: int = 4800,
            activation: Callable[[Tensor], Tensor] = F.gelu,
            use_rotary=True,
            **kwargs
    ) -> None:

        super().__init__()

        self.smpl = smpl
        self.latent_dim = latent_dim

        self.seed_dim = 151

        output_feats = nfeats

        # positional embeddings

        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)

        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim), )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim-self.seed_dim, latent_dim)
        self.seed_projection = nn.Linear(self.seed_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        # FiLM Transformer
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)

        self.final_layer = nn.Linear(latent_dim, output_feats)

        self.refine_input_projection1 = nn.Linear(output_feats, latent_dim)
        self.refine_cond_projection1 = nn.Linear(48, latent_dim)
        self.refine_norm_cond1 = nn.LayerNorm(latent_dim)
        refine_decoderstack = nn.ModuleList([])
        for _ in range(1):
            decoderstack.append(
                FiLMTransformerDecoderLayer2(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.refine_seqTransDecoder1 = DecoderLayerStack(refine_decoderstack)
        self.refine_final_layer1 = nn.Linear(latent_dim, output_feats)

    def get_rcond(self, output ,normalizer):
        # with torch.no_grad():
            if normalizer is not None:
                output = normalizer.unnormalize(output)

            model_contact, model_out = torch.split(output, (4, output.shape[2] - 4), dim=2)

            b, s, c = model_out.shape

            model_x = model_out[:, :, :3]
            model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))

            model_xp = self.smpl.forward(model_q, model_x)
            joints3d = model_xp.reshape(b,s,-1,3)
            # joints3d = do_smplxfk(output, self.smplxfk)[:,:,:22,:]
            B,T,J,_ = joints3d.shape
            l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
            pred_foot = joints3d[:, :, relevant_joints, :]          # B,T,J,4
            foot_vel = torch.zeros_like(pred_foot)
            foot_vel[:, :-1] = (
                pred_foot[:, 1:, :, :] - pred_foot[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            foot_y_ankle = pred_foot[:, :, :2, 1]
            foot_y_toe = pred_foot[:, :, 2:, 1]
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.012)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=3).repeat(1, 1, 1, 3)
            contact_lable = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2).int().to(output).reshape(B, T, -1)

            contact_toe_thresh, contact_ankle_thresh, contact_vel_thresh = -1.2+0.08, -1.2+0.015, 0.3 / 30           # 30 is fps
            contact_score_toe = torch.sigmoid((contact_toe_thresh - pred_foot[:, :, :2, 1])/contact_toe_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, :2, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_toe = torch.unsqueeze(contact_score_toe, dim=3).repeat(1, 1, 1, 3)
            contact_score_ankle = torch.sigmoid((contact_ankle_thresh - pred_foot[:, :, 2:, 1])/contact_ankle_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, 2:, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_ankle = torch.unsqueeze(contact_score_ankle, dim=3).repeat(1, 1, 1, 3)
            contact_score = torch.cat([contact_score_toe, contact_score_ankle], dim = -2).reshape(B, T, -1)
            r_cond = torch.cat([contact_lable, contact_score, pred_foot.reshape(B,T,-1), foot_vel.reshape(B,T,-1)], dim = -1)
            return r_cond

    def guided_forward(self, x, cond_embed, normalizer,times, guidance_weight):
        unc = self.forward(x, cond_embed,normalizer, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, normalizer,times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
            self, x: Tensor, cond_embed: Tensor, normalizer, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)
        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)  # (11,)
        keep_mask_embed = rearrange(keep_mask,
                                    "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask,
                                     "b -> b 1")

        seed = cond_embed[:, :, -151:]
        cond_embed = cond_embed[:, :, :-151]

        seed = seed[:, 0, :]
        seed_embed = self.seed_projection(seed)

        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens,
                                  null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(
            dim=-2)
        cond_hidden = self.non_attn_cond_projection(
            mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(
            t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden,
                                  null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        seed_embed = seed_embed.unsqueeze(1)
        c = torch.cat((cond_tokens,seed_embed, t_tokens),dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens,
                                      t)


        output = self.final_layer(output)

        r_output = self.refine_input_projection1(output)
        r_cond = self.get_rcond(output,normalizer)
        r_cond = self.refine_cond_projection1(r_cond)
        # seed_embed = seed_embed.unsqueeze(1)
        rc = torch.cat((r_cond,t_tokens), dim=-2)  #(11,152,512) 153=150+2
        r_cond_tokens = self.refine_norm_cond1(rc)
        refine_output = self.refine_seqTransDecoder1(r_output, r_cond_tokens, t)
        refine_output = self.refine_final_layer1(refine_output)  # / 10
        out = output + refine_output



        return out
