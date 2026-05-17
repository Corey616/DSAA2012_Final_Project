---
name: sd-diffusers
description: >-
  Documents SDXL UNet attention architecture and Diffusers attn_processors API.
  Explains (1) how self-attention and cross-attention layers are structured in
  SDXL UNet, (2) how to inject custom attention processors via
  pipe.unet.set_attn_processor(), (3) the correct hook points for SCA
  (self-attention K/V concat) vs face embedding (cross-attention feature
  injection). Use this skill when implementing or debugging attention processors.
license: MIT
compatibility:
  - diffusers >= 0.25
metadata:
  priority: high
  last_updated: 2026-05-02
---

# SDXL Diffusers Attention Architecture

## UNet Attention Structure
SDXL UNet has:
- **12 self-attention layers** (6 in down blocks, 6 in up blocks)
- **12 cross-attention layers** (same layout, attend to text encoder hidden states)
- Additional mid-block attention layers

## AttnProcessor API (Diffusers >= 0.25)
```python
class CustomAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, 
                 attention_mask=None, *args, **kwargs):
        # self-attention: encoder_hidden_states = None
        # cross-attention: encoder_hidden_states = text_embeds
        
        # Standard flow:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states if is_self_attn else encoder_hidden_states)
        value = attn.to_v(hidden_states if is_self_attn else encoder_hidden_states)
        
        # SCA (self-attention only): concat K/V with historical frame K/V
        # key = torch.cat([key, historical_key], dim=2)
        # value = torch.cat([value, historical_value], dim=2)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        return hidden_states

# Apply to UNet:
pipe.unet.set_attn_processor(custom_processor)
```

## SCA Injection Points
- **Self-attention only** (cross-attention handles text conditioning)
- Hook into `__call__` method of self-attention processors
- Q stays identical to standard; K/V concatenate features from other frames in batch
- Cross-frame feature dimension: must match UNet hidden state dim (1280 for SDXL)

## Memory Bank Integration
- Memory Bank stores compressed VAE features (128-dim)
- These must be projected to UNet hidden state dimension before SCA injection
- Alternatively, store UNet hidden states directly instead of VAE latents
