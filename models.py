# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


#################################################################################
#              Flash Attention Dispatcher + Custom Attention Layer              #
#################################################################################

_fa2_func = None
_fa3_func = None


def _get_flash_attn_func(op):
    global _fa2_func, _fa3_func
    if op == "fa2":
        if _fa2_func is None:
            from flash_attn_jvp.flash_attention_2_jvp import flash_attn_func
            _fa2_func = flash_attn_func
        return _fa2_func
    if op == "fa3":
        if _fa3_func is None:
            from flash_attn_jvp.flash_attention_3_jvp import flash_attn_func
            _fa3_func = flash_attn_func
        return _fa3_func
    raise ValueError(f"Unsupported flash attention op: {op}")


def attn_op(q, k, v, op="base"):
    """
    Attention operator dispatcher.
    op: "base" | "fa2" | "fa3" | "torch_sdpa"
    Input/output: q, k, v shape (B, L, H, D)
    """
    if op in ("fa2", "fa3"):
        return _get_flash_attn_func(op)(q, k, v)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if op == "torch_sdpa":
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
    elif op == "base":
        scale = q.shape[-1] ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        x = attn.softmax(dim=-1) @ v
    else:
        raise ValueError(f"Unknown attn op: {op}")
    return x.transpose(1, 2)


class FlashAttention(nn.Module):
    """Custom Attention with FA2/FA3/SDPA/base dispatch.
    Weight-compatible with timm Attention (same qkv, proj layout)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_func="torch_sdpa", **kwargs):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = partial(attn_op, op=attn_func)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        x = self.attn(q, k, v)
        x = x.reshape(B, L, C)
        x = self.proj(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        attn_func = block_kwargs.pop("attn_func", None)
        if attn_func is not None:
            self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_func=attn_func)
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        attn_func=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        block_kwargs = {"attn_func": attn_func} if attn_func else {}
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class EndpointConditionProjector(nn.Module):
    """
    Optional global conditioning branch for the dual-stem path model.
    The projector consumes pooled delta-stem tokens and produces an additive
    hidden-size conditioning vector. Its final projection is zero-initialized so
    that loading a vanilla SiT checkpoint still reproduces the teacher at step 0.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, pooled_delta_tokens):
        return self.proj(self.act(self.norm(pooled_delta_tokens)))


class DualStemPathSiT(nn.Module):
    """
    SiT-shaped path network with two spatial stems: one for x_lin(t) and one for
    delta = x1 - x0. The forward signature is (x_lin, delta, t, y).

    Intended use: teacher-residualized path parameterizations, including
        gamma(t) = x_lin(t) + t(1-t) * (path_model(x_lin, delta, t, y) - teacher(x_lin, t, y))
    and the subtraction-based boundary variant
        gamma(t) = x_lin(t) + c(t) - (1-t) c(0) - t c(1),
    where c(t) = path_model(x_lin(t), delta, t, y) - teacher(x_lin(t), t, y).

    With x_embedder_delta and the optional endpoint-conditioning branch zero-initialized,
    and all shared SiT weights copied from a pretrained teacher, the model exactly
    reproduces the teacher output at initialization. For the subtraction-based
    path, that implies c(t) = c(0) = c(1) = 0 and therefore gamma(t) = x_lin(t).
    """

    uses_dual_stem_path = True
    uses_teacher_residualized_path = True

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_endpoint_conditioning=True,
        teacher_residual_boundary_lambda=1.0,
        attn_func=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_endpoint_conditioning = use_endpoint_conditioning
        self.teacher_residual_boundary_lambda = teacher_residual_boundary_lambda

        self.x_embedder_lin = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder_delta = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder_lin.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if use_endpoint_conditioning:
            self.endpoint_conditioner = EndpointConditionProjector(hidden_size)
        else:
            self.endpoint_conditioner = None

        block_kwargs = {"attn_func": attn_func} if attn_func else {}
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder_lin.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w_lin = self.x_embedder_lin.proj.weight.data
        nn.init.xavier_uniform_(w_lin.view([w_lin.shape[0], -1]))
        nn.init.constant_(self.x_embedder_lin.proj.bias, 0)

        # Crucial for exact linear-path initialization after copying teacher weights.
        self.zero_new_path_modules()

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def zero_new_path_modules(self):
        nn.init.constant_(self.x_embedder_delta.proj.weight, 0)
        nn.init.constant_(self.x_embedder_delta.proj.bias, 0)
        if self.endpoint_conditioner is not None:
            nn.init.constant_(self.endpoint_conditioner.proj.weight, 0)
            nn.init.constant_(self.endpoint_conditioner.proj.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder_lin.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _build_condition(self, t, y, delta_tokens):
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        if self.endpoint_conditioner is not None:
            pooled_delta_tokens = delta_tokens.mean(dim=1)
            c = c + self.endpoint_conditioner(pooled_delta_tokens)
        return c

    def forward(self, x_lin, delta, t, y):
        x_lin_tokens = self.x_embedder_lin(x_lin)
        x_delta_tokens = self.x_embedder_delta(delta)
        x = x_lin_tokens + x_delta_tokens + self.pos_embed
        c = self._build_condition(t, y, x_delta_tokens)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_from_endpoints(self, x0, x1, t, y):
        t_view = t.view(-1, 1, 1, 1)
        x_lin = (1.0 - t_view) * x0 + t_view * x1
        delta = x1 - x0
        return self.forward(x_lin, delta, t, y)

    def forward_with_cfg(self, x_lin, delta, t, y, cfg_scale):
        half_x_lin = x_lin[: len(x_lin) // 2]
        half_delta = delta[: len(delta) // 2]
        combined_x_lin = torch.cat([half_x_lin, half_x_lin], dim=0)
        combined_delta = torch.cat([half_delta, half_delta], dim=0)
        model_out = self.forward(combined_x_lin, combined_delta, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def _rename_x_embedder_key_for_dual_stem(key):
    if key.startswith("x_embedder."):
        return "x_embedder_lin." + key[len("x_embedder.") :]
    return key


def is_dual_stem_path_state_dict(state_dict):
    return "x_embedder_lin.proj.weight" in state_dict and "x_embedder_delta.proj.weight" in state_dict


def load_dual_stem_path_from_sit_state_dict(model, state_dict, strict_backbone=True, zero_init_new_modules=True):
    if not isinstance(model, DualStemPathSiT):
        raise TypeError("load_dual_stem_path_from_sit_state_dict expects a DualStemPathSiT instance.")

    remapped_state_dict = {
        _rename_x_embedder_key_for_dual_stem(key): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(remapped_state_dict, strict=False)

    expected_missing = {
        "x_embedder_delta.proj.weight",
        "x_embedder_delta.proj.bias",
    }
    if model.endpoint_conditioner is not None:
        expected_missing.update({
            "endpoint_conditioner.proj.weight",
            "endpoint_conditioner.proj.bias",
        })

    if strict_backbone:
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        extra_missing = missing - expected_missing
        if unexpected:
            raise ValueError(f"Unexpected keys when loading vanilla SiT weights into DualStemPathSiT: {sorted(unexpected)}")
        if extra_missing:
            raise ValueError(f"Missing backbone keys when loading vanilla SiT weights into DualStemPathSiT: {sorted(extra_missing)}")

    if zero_init_new_modules:
        model.zero_new_path_modules()
    return incompatible


def load_dual_stem_path_state_dict(model, state_dict, strict=True, zero_init_new_modules=True):
    if is_dual_stem_path_state_dict(state_dict):
        return model.load_state_dict(state_dict, strict=strict)
    return load_dual_stem_path_from_sit_state_dict(
        model,
        state_dict,
        strict_backbone=strict,
        zero_init_new_modules=zero_init_new_modules,
    )


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SIT_MODEL_CONFIGS = {
    'SiT-XL/2': dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    'SiT-XL/4': dict(depth=28, hidden_size=1152, patch_size=4, num_heads=16),
    'SiT-XL/8': dict(depth=28, hidden_size=1152, patch_size=8, num_heads=16),
    'SiT-L/2':  dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    'SiT-L/4':  dict(depth=24, hidden_size=1024, patch_size=4, num_heads=16),
    'SiT-L/8':  dict(depth=24, hidden_size=1024, patch_size=8, num_heads=16),
    'SiT-B/2':  dict(depth=12, hidden_size=768,  patch_size=2, num_heads=12),
    'SiT-B/4':  dict(depth=12, hidden_size=768,  patch_size=4, num_heads=12),
    'SiT-B/8':  dict(depth=12, hidden_size=768,  patch_size=8, num_heads=12),
    'SiT-S/2':  dict(depth=12, hidden_size=384,  patch_size=2, num_heads=6),
    'SiT-S/4':  dict(depth=12, hidden_size=384,  patch_size=4, num_heads=6),
    'SiT-S/8':  dict(depth=12, hidden_size=384,  patch_size=8, num_heads=6),
}


def build_dual_stem_path_sit(model_name, **kwargs):
    if model_name not in SIT_MODEL_CONFIGS:
        raise KeyError(f"Unknown SiT model variant: {model_name}")
    return DualStemPathSiT(**SIT_MODEL_CONFIGS[model_name], **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
