# Transformer-based UNet 替代方案研究

## 背景
TA 反馈建议："考虑将 UNet 骨干网络替换为基于 Transformer 的原型。"

## 现有 Transformer 扩散架构

### 1. DiT (Diffusion Transformer) — Peebles & Xie, 2023
- **论文**: "Scalable Diffusion Models with Transformers" (arXiv:2212.09748)
- **架构**: 用 Vision Transformer (ViT) 替换 UNet，在 ImageNet 上达到 SOTA
- **设计**: 将输入处理为 patch 序列，使用 adaLN (自适应层归一化) 替代 UNet 的时间步长/类别嵌入
- **优缺点**: 扩展性好，但生成视觉上丰富的场景需要参数密集型模型

### 2. PixArt-α / PixArt-Σ — Huawei, 2024
- **架构**: 构建在 DiT 之上，专为文本生成图像进行效率优化。使用交叉注意力模块注入文本条件。
- **关键特征**: 比 SDXL 训练速度更快（训练时间仅需 10.8%）且推理速度更快（内存的 22%）
- **质量**: 在 FID 和 CLIP 评分方面，与 SDXL 同等或更优
- **与我们的适配性**: 通过 diffusers 提供原生 Pipeline。支持交叉注意力。需对 SCA 代码进行适配。

### 3. MMDiT (Multi-Modal Diffusion Transformer) — SD3 / Flux
- **架构**: 在单个体内联合处理图像和文本 token
- **质量**: SD3 在图像质量上显著优于 SDXL
- **限制**: 模型更大（8B vs SDXL 的 2.6B），需要更多 GPU 显存

### 4. Flux — Black Forest Labs, 2024
- **架构**: 基于 MMDiT，进一步优化，整合了流匹配和旋转位置编码 (RoPE)
- **质量**: 当前在图像质量方面处于领先地位
- **与我们的适配性**: 与 SDXL 的推理接口不同（使用 T5 文本编码器 + CLIP）。需要对 SCA 进行全面重新实现。

## 与 StoryGen 的适配性分析

| 架构 | 易于集成 | 图像质量 | 显存需求 | 最大分辨率 |
|----------|--------------|-------------|-----------|-----------------|
| SDXL (当前) | ✅ 当前 | 基准 | 中等 (2.6B) | 1024×1024 |
| PixArt-Σ | ⚠️ 需要适配 | ✅ 优于 SDXL | 更低 (0.6B) | 1024+ |
| Flux (MMDiT) | ❌ 难度大 | ✅✅ 最优 | 高 (8B) | 1024×1024 |
| DiT | ⚠️ 需要适配 | 与 SDXL 相当 | 取决于架构 | 可变 |

## SCA 兼容性分析

当前 SCA 处理器修改了 SDXL UNet 的自注意力模块 (attn1)。Transformer 扩散模型同样使用自注意力模块，这使得理论上的适配成为可能。关键差异：
1. **Transformer 架构使用标准 Transformer 块**而非 UNet 的下采样/上采样阶段
2. **层的标准化方式不同** (adaLN vs GroupNorm)
3. **嵌入文本的方式不同** (交叉注意力 vs CLIP/text encoder 连接)

### 适配策略
1. 将 `ConsistentSelfAttentionProcessor` 适配为与 PixArt-Σ 的自注意力兼容
2. 使用 PixArt-Σ 的 pipe（`PixArtSigmaPipeline`）替换 `StableDiffusionXLPipeline`
3. 调整 `NarrativeGenerationPipeline`，使其使用 PixArt 而非 SDXL
4. 如 diffusers 的兼容性或显存限制不允许，则回退至 DiT

---

## 衣物一致性研究

对于多角色场景中的衣物颜色/样式不一致，最可行的方案包括：

### 训练时方案
1. **Storybooth 有界跨帧注意力** — 需要自定义交叉注意力处理器，该处理器要暴露注意力概率。我们已有的 `set_bounded_masks()` 基础设施已就位，但缺少掩码生成模型。
2. **角色特定的令牌嵌入** — 向 UNet 注入角色身份嵌入。由于训练量庞大且需数据集支持，目前不现实。

### 训练时方案
1. **CLIP 文本编码器提示增强** — 使用 LLM 生成高度具体的衣物描述（效果已获验证，参见第 4 轮，+0.022）。
2. **针对性 SCA 强度调度** — 对处理具体细节（如衣物）的 up_blocks 降低 SCA 强度。我们当前的层强度为 up=0.07，已非常低。
3. **通过 LLM 规则加强场景分离** — 多角色故事需在系统提示中加入额外规则，以对每个角色使用不同场景。

### 推荐路径
1. 维持 LLM 系统提示的改进（已获验证，效果积极）。
2. 实施自定义交叉注意力处理器，以支持 Storybooth 风格的有界掩码。
3. 评估 PixArt-Σ 作为 SDXL 的替代后端——可能带来更好的出生质量，同时对 SCA 的影响更易控制。
