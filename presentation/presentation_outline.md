# 演讲分工安排

## 演讲顺序

| 顺序 | 演讲人 | 内容 | 时间 |
|------|--------|------|------|
| 1 | **Keyu** | 开场 + Pipeline 概述 | ~3 min |
| 2 | **Siqi** | LLM Parser 部分 | ~5 min |
| 3 | **Zhenzhuo** | SDXL / Core Generator 部分 | ~5 min |
| 4 | **Keyu** | Evaluation 部分 | ~4 min |
| 5 | **Zhenzhuo** | 成功案例展示 | ~3 min |
| 6 | **Siqi** | 局限性分析 | ~3 min |
| 7 | **Keyu** | 总结 + Q&A | ~2 min |

---

## 详细分工

### Keyu (开场 + Pipeline + Evaluation + 总结)

**开场介绍 (Slide 1-2)**
- 自我介绍
- 项目概述：从文本脚本到多帧图像生成的故事系统

**Pipeline 概述 (Slide 3-4)**
- 四层架构介绍：Script Director → Asset Anchor → Core Generator → Evaluation Hub
- 整体数据流：从 `[SCENE-N]` 标签到 storyboard 输出的完整流程

**Evaluation 部分 (Slide 12-14)**
- CLIP Score 指标解释（文本-图像对齐度）
- LPIPS Consistency 指标解释（帧间身份一致性）
- 定量结果表格解读
- Overall Score 计算公式

**总结 + Q&A (Slide 15)**
- 核心贡献总结
- 引导 Q&A 环节

---

### Siqi (LLM Parser 部分)

**Phase 1: Story Analysis (Slide 5)**
- 角色识别与唯一 ID 分配（`[Lily_001]`）
- Visual anchor 提取：服装、发型、配色
- 动作分解到每一帧

**Phase 2: Prompt Generation (Slide 6)**
- 基于 Phase 1 分析生成 SDXL 提示词
- 跨帧外观一致性强制
- 多角色场景的 LEFT/RIGHT 空间定位

**Phase 3: Refinement (Slide 7)**
- 一致性验证（服装漂移检测）
- 代词消解（he/she/they → 具体角色名）
- Visual description 前缀匹配进行角色去重

**局限性分析 (Slide 10-11)**
- 非人类角色的处理挑战
- 抽象实体（如 Robot）的识别问题

---

### Zhenzhuo (SDXL / Core Generator + 案例展示)

**Core Generator 架构 (Slide 4 + Backup Slide 17)**
- SDXL Pipeline 懒加载机制
- Character Portrait 生成流程
- Memory Bank 压缩视觉记忆库

**成功案例展示 (Slide 8-9)**
- Case 11: Olivia - 优秀的一致性表现
- Case 01: Lily - 强面部身份保持
- Case 12: Ethan - 场景连贯性

---

## 关键演讲要点

### Keyu 必讲要点
1. "我们的系统通过四层架构实现从文本到图像的自动化生成"
2. "CLIP Score ≥ 0.25，LPIPS Consistency ≥ 0.30，这是我们的评估标准"
3. "32个测试案例100%成功生成"

### Siqi 必讲要点
1. "LLM Parser 分为三个阶段：故事分析、提示词生成、优化"
2. "Visual description 是角色的唯一真相源"
3. "代词消解解决了'两个Milo'的问题"

### Zhenzhuo 必讲要点
1. "SDXL Pipeline 采用懒加载策略优化显存"
2. "Memory Bank 将特征压缩到128维，通过重要性加权检索"
3. "Character Portrait 提供角色视觉锚点"

---

## 演讲时间控制

| 部分 | 建议时间 | 实际时间 |
|------|----------|----------|
| 开场 + Pipeline 概述 | 3 min | |
| LLM Parser | 5 min | |
| SDXL / Core Generator | 5 min | |
| Evaluation | 4 min | |
| 案例展示 | 3 min | |
| 局限性分析 | 3 min | |
| 总结 + Q&A | 2 min | |
| **总计** | **~25 min** | |

---

## 演示时的配合建议

1. **换人时**：下一位直接接着讲，不需要过多过渡
2. **案例展示时**：Zhenzhuo 可以指着图像说"看这里..."增强说服力
3. **Q&A 时**：Keyu 主持，其他人补充相关技术细节
