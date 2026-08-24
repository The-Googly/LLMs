# llm_evolution

**An evolution tree of large language models — Transformer (2017) → the 2026 frontier.**

This doc is a *graph*, not a history essay. Nodes are architectures, mechanisms, or training regimes. **Edges are the design decision that got you from one to the next.** Read the edges — that is where the actual story lives.

Companion file: `llm_docs.md` (reference material, read-only). Where `llm_docs` already explains a mechanism in depth, this doc links to it rather than repeating it.

---

<a id="index"></a>

## Index

1. [How to read this doc](#how-to-read)
2. [The master graph](#master-graph)
3. [Branch 0 — Ancestry (pre-2017)](#b0)
4. [Branch 1 — The trunk: Transformer 2017](#b1)
5. [Branch 2 — The great split: encoder vs decoder](#b2)
6. [Branch 3 — Scale and the compute-allocation correction](#b3)
7. [Branch 4 — The block refit (RoPE / RMSNorm / SwiGLU)](#b4)
8. [Branch 5 — Position and context length](#b5)
9. [Branch 6 — The KV-cache war (MQA → GQA → MLA → sparse → linear)](#b6)
10. [Branch 7 — Systems, not math (FlashAttention and friends)](#b7)
11. [Branch 8 — Conditional compute (MoE)](#b8)
12. [Branch 9 — Alignment and post-training](#b9)
13. [Branch 10 — The reasoning / RLVR era](#b10)
14. [Branch 11 — Multimodality and fusion](#b11)
15. [Branch 12 — Topology: rethinking the residual stream (2026)](#b12)
16. [Branch 13 — Optimizers and numerics](#b13)
17. [Where the frontier stands (2026 convergence table)](#convergence)
18. [Open branches — unresolved bets](#open)
19. [Appendix — deep dives](#appendix)
20. [Sources](#sources)

---

<a id="how-to-read"></a>

## How to read this doc

Every node follows the same four-line schema:

```
### <Name> (<year>) — <paper / lab>
Parent:  ← what it descends from
Edge:    the one design change made at this step
Why:     the problem that change solves
Impact:  what it unlocked / who adopted it
```

Conventions:

- `←` means "descends from". `⇄` means "competing sibling, same problem".
- **✦** marks a node I consider a genuine branch point — a change that forced everyone downstream to pick a side. (Same legend spirit as `llm_docs`.)
- `[→ A.n]` links to a deep dive in the [Appendix](#appendix).
- A node's *edge* is deliberately short. If you want mechanism, go to the appendix or to `llm_docs`.
- **Dates are paper/release dates**, not adoption dates. Adoption usually lags 6–18 months.

One structural warning: this is a **graph, not a tree**. Modern models are *merges* — Kimi K3 inherits MoE from Shazeer-1991-via-DeepSeekMoE, attention from the DeltaNet line, normalization from RMSNorm, optimizer from Muon, and residual topology from its own AttnRes work. Four ancestries in one block.

---

<a id="master-graph"></a>

## The master graph

The trunk, top to bottom. Horizontal spread = competing answers to the same problem.

```
                 RNN → LSTM → seq2seq (2014)
                          │
                 Bahdanau attention (2014)  ── "stop cramming a sentence into one vector"
                          │
                 ══════════════════════════════
                 TRANSFORMER (2017)  ✦  "drop recurrence entirely"
                 ══════════════════════════════
                          │
              ┌───────────┴────────────┐
              │                        │
        encoder-only              decoder-only  ✦
        BERT (2018)               GPT-1 (2018)
        RoBERTa, DeBERTa          GPT-2 (2019)   ── Pre-LN, byte BPE, zero-shot
        ModernBERT (2024)         GPT-3 (2020)   ── in-context learning
              │                        │
        (embeddings, rerankers,        │
         encoders inside VLMs)         │
                                       ▼
                      ┌────────────────┴─────────────────┐
                      │                                  │
             Kaplan scaling laws (2020)          the block refit (2019-22)
                      │                          RoPE · RMSNorm · SwiGLU
             Chinchilla (2022) ✦                          │
             "you were undertrained"                      ▼
                      └───────────────► LLaMA (2023) ✦ ◄──┘
                                   the canonical open block
                                            │
        ┌──────────────┬───────────────┬────┴─────────┬──────────────┬──────────────┐
        ▼              ▼               ▼              ▼              ▼              ▼
    KV CACHE       CONDITIONAL      SYSTEMS      POST-TRAINING   POSITION /     TOPOLOGY
    (Branch 6)     COMPUTE          (Branch 7)   (Branch 9-10)   CONTEXT        (Branch 12)
        │          (Branch 8)           │              │         (Branch 5)         │
    MQA (2019)     Shazeer MoE      FlashAttn      RLHF (2022)   ALiBi/NoPE     Hyper-Conn
        │          (2017)           1/2/3            │          PI · YaRN       (2024)
    GQA (2023)         │            (2022-24)      DPO (2023)   SWA rings          │
        │          Switch (2021)        │              │            │           mHC (2025)
    MLA (2024) ✦       │            vLLM Paged     RLVR/GRPO       │               │
        │          DeepSeekMoE ✦    (2023)         (2024-25) ✦     │           AttnRes
        ├──────────(2024)               │              │           │           (2026) ✦
        │          fine-grained     spec. decode   reasoning /      │
   sparse attn     + shared         + MTP          thinking modes   │
   NSA (2025)          │                                            │
   DSA  (2025) ✦   aux-loss-free                                    │
        │          balancing                                        │
   linear/hybrid       │                                            │
   GLA · DeltaNet  LatentMoE (2026)                                 │
   GatedDeltaNet ✦     │                                            │
   KDA (2025) ✦        │                                            │
        └──────────────┴──────────────┬─────────────────────────────┘
                                      ▼
                    ═══════════════════════════════════════
                            THE 2026 FRONTIER
                    ═══════════════════════════════════════
      DeepSeek V4        Kimi K3          Qwen3.5        Gemma 4       GLM-5
      1.6T / 49B-A       2.8T / 16-of-896  397B / 17B-A   2.3B–31B      MLA+DSA
      CSA+HCA sparse     KDA + Gated MLA   GDN:full 3:1   dense+MoE
      mHC topology       AttnRes + NoPE    gated attn     5:1 SWA
      1M ctx             LatentMoE, MXFP4  262K ctx       encoder-free MM
```

The single most important thing this picture shows: after ~2023 the trunk stops being about *capability* and becomes about **cost per unit of capability**. Almost every 2024–2026 edge is an efficiency edge — memory, FLOPs, or bits — with quality held constant or slightly improved.

---

<a id="b0"></a>

## Branch 0 — Ancestry (pre-2017)

### seq2seq with RNN/LSTM (2014) — Sutskever et al.

- **Parent:** ← statistical MT / RNN LMs
- **Edge:** encode a whole source sentence into one fixed-size vector, decode from it.
- **Why:** first end-to-end neural translation without alignment machinery.
- **Impact:** established encoder→decoder as the shape. Also established its own fatal flaw: one vector is an information bottleneck.

### ✦ Bahdanau attention (2014) — Bahdanau, Cho, Bengio

- **Parent:** ← seq2seq
- **Edge:** instead of one context vector, let the decoder compute a **weighted sum over all encoder states**, weights learned per output step.
- **Why:** the bottleneck. Long sentences degraded badly.
- **Impact:** this is the ancestor of every line in this document. Everything after 2017 is a variation on "weighted sum over other positions, weights from a learned similarity."

### ELMo (2018) — Peters et al.

- **Parent:** ← biLSTM LMs
- **Edge:** *contextual* word vectors from a pretrained bidirectional LM, fed into task models.
- **Why:** word2vec/GloVe give one vector per word type; "bank" needs two.
- **Impact:** proved "pretrain a LM, transfer the representations". GPT-1 and BERT both take this idea and drop the LSTM.

---

<a id="b1"></a>

## Branch 1 — The trunk

### ✦ Transformer (2017) — Vaswani et al., *Attention Is All You Need*

- **Parent:** ← Bahdanau attention + seq2seq
- **Edge:** **delete recurrence.** Replace it with self-attention everywhere; add multi-head projections, sinusoidal positions, residual+LayerNorm wrappers.
- **Why:** RNNs are sequential — O(N) unparallelizable steps per sequence. Attention makes every position reachable in O(1) hops and the whole sequence parallel over time.
- **Impact:** the entire field. Also introduced four sub-decisions that each became their own branch later: scaled dot-product (√dₖ), multi-head, residual+norm placement, and additive positional encoding.

> Full mechanism in `llm_docs` → [Transformer 2017](llm_docs.md#transformer-2017). Not repeated here.

The four inherited sub-decisions and where each one got overturned:

| 2017 decision | Overturned by | Branch |
|---|---|---|
| Sinusoidal, **added** to embeddings | RoPE — multiplicative, relative | [B5](#b5) |
| **Post-LN** (`LN(x + F(x))`) | Pre-LN (GPT-2) | [B2](#b2) |
| LayerNorm | RMSNorm | [B4](#b4) |
| Every head gets own K/V | MQA → GQA → MLA → latent/linear | [B6](#b6) |
| Dense FFN, ReLU | SwiGLU, then MoE | [B4](#b4), [B8](#b8) |
| Uniform residual accumulation | Hyper-connections, AttnRes | [B12](#b12) |

---

<a id="b2"></a>

## Branch 2 — The great split: encoder vs decoder

The 2017 model had both stacks. Within a year the field split on **which half to keep**, and the split was really about *what objective you can train at scale*.

### BERT (2018) — Devlin et al. — the encoder branch

- **Parent:** ← Transformer encoder
- **Edge:** keep the encoder only; train with **masked LM** (bidirectional) instead of next-token.
- **Why:** for *understanding* tasks, seeing both directions is strictly more information than causal.
- **Impact:** dominated NLP benchmarks 2018–2020. But MLM is not generative and does not benefit from scale the same way — this branch plateaued as a *general* model and survives today as embedding models, rerankers, and the vision/audio encoders bolted onto multimodal LLMs. `ModernBERT` (2024) is the branch's modern refresh: same objective, modern block (RoPE, GeGLU, unpadding).

### ✦ GPT-1 (2018) — Radford et al. — the decoder branch

- **Parent:** ← Transformer decoder
- **Edge:** drop the encoder *and* the cross-attention sublayer. Block = masked self-attention + FFN. Pipeline = generative pretrain, then supervised fine-tune, **same weights**.
- **Why:** no source sentence exists in a plain LM. Causal LM needs no labels, so pretraining data is effectively unlimited.
- **Impact:** the winning branch. [→ `llm_docs`](llm_docs.md#gpt-1)

### GPT-2 (2019) — Radford et al.

- **Parent:** ← GPT-1
- **Edge:** three depth-enabling fixes — **Pre-LN**, **1/√N residual init**, **final LN before unembedding** — plus **byte-level BPE** and 10× scale (1.5B, 48 layers).
- **Why:** Post-LN gradients are depth-unstable and need warmup babysitting; byte BPE removes OOV entirely.
- **Impact:** Pre-LN is still the default in 2026. Byte-BPE is still the default tokenizer family. Also the first serious claim that a plain LM does tasks **zero-shot**. [→ `llm_docs`](llm_docs.md#gpt-2)

### ✦ GPT-3 (2020) — Brown et al.

- **Parent:** ← GPT-2
- **Edge:** 175B params, 96 layers, 2048 ctx, alternating dense/**locally-banded sparse** attention. No architectural novelty of note — the edge is **scale plus a reframing**.
- **Why:** if zero-shot works a bit, does more scale make fine-tuning unnecessary?
- **Impact:** **in-context learning.** The deliverable became a prompt, not a fine-tune. This is what turned "language model" into "product", and it is why every branch below is measured in *inference cost* — inference suddenly mattered more than training.

---

<a id="b3"></a>

## Branch 3 — Scale and the compute-allocation correction

### Kaplan scaling laws (2020) — OpenAI

- **Parent:** ← GPT-3 era empiricism
- **Edge:** loss follows smooth power laws in params, data, and compute. Given fixed compute, **spend most of it on parameters**.
- **Why:** make model design a budgeting problem instead of guesswork.
- **Impact:** justified the 2020–2021 parameter race (Gopher 280B, MT-NLG 530B). It was also **wrong about the ratio**, because of a learning-rate schedule artifact.

### ✦ Chinchilla (2022) — Hoffmann et al., DeepMind

- **Parent:** ← Kaplan
- **Edge:** redo the sweep properly. Optimal is roughly **params and tokens scaled equally** (~20 tokens/param), not param-heavy.
- **Why:** Kaplan's setup didn't re-tune the optimizer per model size. Porian et al. (2024) later isolated the three causes, in order: (1) not tuning **learning rate, batch size and AdamW β₂ per scale** — this alone moves the exponent from ~0.60 to ~0.50; (2) a **constant-length warmup** that is far too long for small models; (3) **excluding the output head** from the FLOP count.
- **Impact:** ✦ the biggest single course correction in the field. 70B Chinchilla beat 280B Gopher. Killed the naive parameter race and directly produced LLaMA. **Second-order effect that matters more:** once you account for *inference* cost, you train **past** Chinchilla-optimal on purpose — small model, absurd token count — which is exactly the LLaMA/Qwen/Gemma recipe and why a 8B model in 2026 is usable at all.

### Overtraining and distillation as the small-model recipe (2023→)

- **Parent:** ← Chinchilla, inverted
- **Edge:** train small models on 10–40× Chinchilla tokens; additionally **distill** from a large teacher (logit-level).
- **Why:** for a model you will serve billions of times, training compute is a rounding error against inference compute.
- **Impact:** the entire small-open-model category (LLaMA 7B, Phi, Gemma, Qwen small, SmolLM). Gemma has used distillation as the primary recipe since Gemma 2.

---

<a id="b4"></a>

## Branch 4 — The block refit

Between 2019 and 2022, three independent papers each replaced one component of the 2017 block. By 2023 all three had merged into a single canonical block that essentially every open model still uses.

### RMSNorm (2019) — Zhang & Sennrich

- **Parent:** ← LayerNorm
- **Edge:** drop mean-centering and the β bias; normalize by RMS only.
- **Why:** re-scaling stabilizes training, re-centering contributes ~nothing. One pass instead of two.
- **Impact:** 10–40% faster per norm call, and Pre-LN calls a norm twice per layer. Free win, universal adoption. [→ `llm_docs`](llm_docs.md#rmsnorm)

### SwiGLU (2020) — Shazeer

- **Parent:** ← ReLU/GELU FFN
- **Edge:** gated FFN — one projection **gates** another elementwise; Swish as the gate.
- **Why:** a learned, input-dependent, per-dimension gate beats one fixed nonlinearity. (Shazeer's own justification: it just works better.)
- **Impact:** standard FFN in LLaMA, PaLM, Mistral, Qwen, Gemma. Costs a third matrix; hidden width shrunk to compensate. [→ `llm_docs`](llm_docs.md#swiglu)

### ✦ RoPE (2021) — Su et al., *RoFormer*

- **Parent:** ← sinusoidal / learned absolute positions
- **Edge:** stop *adding* position. **Rotate** Q and K by an angle proportional to absolute index; rotation composition makes the dot product depend only on the **relative** offset.
- **Why:** attention routes by dot product, and routing should depend on distance, not on where in the document a pair sits. Absolute embeddings only approximate this; RoPE gets it exactly, with zero extra parameters.
- **Impact:** ✦ near-total adoption. Also the *enabler* of context extension — because RoPE is a continuous function of position, you can interpolate/rescale it post-hoc (PI, NTK, YaRN) to stretch a 4k model to 128k. That was not possible with a learned position table. [→ `llm_docs`](llm_docs.md#rope)

### ✦ LLaMA (Feb 2023) — Touvron et al., Meta — the merge node

- **Parent:** ← GPT-3 shape ⊕ Chinchilla budgeting ⊕ RoPE ⊕ RMSNorm ⊕ SwiGLU
- **Edge:** not a new idea — a **canonicalization**. Pre-LN decoder, RMSNorm, SwiGLU, RoPE, no biases, heavily overtrained, open weights.
- **Why:** demonstrate that the public recipe, done carefully, reaches GPT-3.5 territory at 13–65B.
- **Impact:** ✦ this is the single most-forked node in the graph. Mistral, Qwen, Yi, DeepSeek, Gemma, OLMo, Phi all start from this block and differ in what they *replace* in it. Everything below Branch 5 is best read as "what did you swap out of the LLaMA block."

### QK-Norm (2020, adopted 2024→) — Henry et al.

- **Parent:** ← the LLaMA block
- **Edge:** RMSNorm applied to Q and K **before** the dot product.
- **Why:** at scale, attention logits drift to huge magnitudes and blow up training (loss spikes). Normalizing Q/K caps the logit scale directly.
- **Impact:** now standard for stability: OLMo 2, Gemma 3/4, Qwen3, MiniMax M2. It is the *architectural* answer to the same problem MuonClip answers on the *optimizer* side ([B13](#b13)).

---

<a id="b5"></a>

## Branch 5 — Position and context length

Once RoPE existed, "how long a context can this model handle" became a separate, actively-contested branch.

### Sliding-window attention (2020 Longformer → 2023 Mistral → Gemma)

- **Parent:** ← full attention
- **Edge:** most layers attend only to a local window (e.g. 1024–4096 tokens); a minority attend globally.
- **Why:** attention is quadratic and most heads are local anyway. KV cache for a window layer is **constant** in sequence length.
- **Impact:** Gemma 3/4 run a **5:1 local:global ratio**; gpt-oss interleaves similarly. Cheapest possible long-context trick, and it composes with everything else.

### Position interpolation → NTK-aware → YaRN (2023)

- **Parent:** ← RoPE
- **Edge:** rescale RoPE frequencies (uniformly, then per-frequency-band) so a model trained at 4k can be fine-tuned briefly to 32k–128k.
- **Why:** extrapolating RoPE past training length fails; interpolating inside the trained range does not.
- **Impact:** the standard "context extension" stage in every long-context recipe. Gemma 4's **pp-RoPE (p=0.25)** on global layers, and its split base frequencies (**1M global / 10k local**), are descendants of this idea.

### ALiBi (2021) ⇄ NoPE (2023)

- **Parent:** ← RoPE, as competing siblings
- **Edge:** ALiBi — no position embedding, just a linear distance penalty added to attention logits. NoPE — **no positional signal at all**; causal masking alone lets the model infer position.
- **Why:** both target length extrapolation. ALiBi extrapolates gracefully by construction; NoPE tests whether explicit position is needed at all.
- **Impact:** ALiBi largely lost to RoPE+YaRN. **NoPE, surprisingly, won a comeback** — used on a subset of layers in SmolLM3, Kimi Linear, and (per Moonshot's own notes) **Kimi K3 replaces RoPE with NoPE outright**, relying on the linear-attention layers to carry positional information recurrently. Watch this one.

### Llama 4 iRoPE (2025)

- **Parent:** ← RoPE ⊕ NoPE ⊕ SWA
- **Edge:** interleave RoPE local layers with **NoPE global** layers, plus inference-time attention-temperature scaling.
- **Why:** a 10M-token target context; RoPE global layers do not extrapolate that far.
- **Impact:** the clearest public statement of the emerging consensus — **position is a per-layer choice, not a model-wide one.**

---

<a id="b6"></a>

## Branch 6 — The KV-cache war

The busiest branch in the graph. One problem, six answers.

**The problem:** at decode time, K and V for every past token must be kept. Cost = `2 · layers · heads · d_head · seq_len` per sequence. At 128k context this exceeds the *weights*, and it caps batch size, which is what actually determines serving cost. Every node below trades some part of attention's expressiveness for cache bytes.

```
                     MHA (2017)  — h heads, h K/V
                          │
        ┌─────────────────┼──────────────────────┬────────────────────┐
        │                 │                      │                    │
   share K/V         compress K/V           attend to less        change the operator
        │                 │                      │                    │
   MQA (2019)        MLA (2024) ✦           SWA (2020)          linear attn (2020)
   1 shared K/V      latent c^KV,           NSA (2025)          GLA (2023)
        │            up-project             DSA (2025) ✦        DeltaNet (2021/24)
   GQA (2023) ✦      per head               CSA+HCA (2026)      Gated DeltaNet ✦
   g groups               │                      │              Mamba-2 (2024)
        │                 │                      │              KDA (2025) ✦
        └─────────────────┴──────────────────────┴────────────────────┘
                                      │
                            hybrid stacks (2025-26)
                       "use different attention per layer"
```

### MQA (2019) — Shazeer

- **Parent:** ← MHA
- **Edge:** keep h query heads, share **one** K and **one** V across all of them.
- **Why:** cache shrinks by exactly h (32× at h=32).
- **Impact:** too aggressive alone — measurable quality loss. But it defined the axis. [→ `llm_docs`](llm_docs.md#mqa-gqa)

### ✦ GQA (2023) — Ainslie et al.

- **Parent:** ← MQA ⇄ MHA
- **Edge:** g groups of query heads, one K/V per group. `g=1` is MQA, `g=h` is MHA.
- **Why:** MQA gave up too much; MHA saves nothing. Make it a dial.
- **Impact:** the default for two years — LLaMA 2/3, Mistral, Qwen, Gemma. Still the baseline any new attention must beat.

### ✦ MLA — Multi-head Latent Attention (2024) — DeepSeek-V2

- **Parent:** ← GQA
- **Edge:** stop *sharing*, start **compressing**. Project the hidden state down to a low-rank latent `c^KV`, **cache only that**, up-project to full per-head K and V at use time. Queries stay full multi-head. RoPE dims are carried in a separate small "decoupled" slice.
- **Why:** GQA sacrifices head diversity to save bytes. MLA keeps per-head K/V *distinct* and pays instead in a rank bottleneck — empirically a much better trade.
- **Impact:** ✦ DeepSeek-V2/V3/R1, Kimi K2/K2.5, GLM-4.5/5. The most consequential attention change since GQA, and the reason DeepSeek's inference economics were a shock in 2024–25. [→ `llm_docs`](llm_docs.md#mla), [→ A.1](#a1)

### NSA → ✦ DSA — DeepSeek sparse attention (2025)

- **Parent:** ← MLA
- **Edge:** **NSA** (Feb 2025): natively-trainable sparse attention — a compressed-token branch, a selected-token branch, and a sliding window, combined by a learned gate. **DSA** (V3.2-Exp, Sept 2025): a cheap **"lightning indexer"** scores all past tokens and each query attends to only its **top-k** (k≈2048), on top of MLA.
- **Why:** MLA fixes cache *size* but attention is still O(N²) in FLOPs. Sparsity attacks the compute, not the memory.
- **Impact:** roughly 2–3× cheaper long-context serving at benchmark parity. Adopted beyond DeepSeek — **GLM-5 ships MLA+DSA**. This is the node where "attend to everything" stopped being assumed.

### DeepSeek V4: CSA + HCA (2026)

- **Parent:** ← MLA ⊕ DSA
- **Edge:** replace MLA with two **interleaved** compressed-sparse mechanisms: **CSA** (softmax-gated pooling compresses KV 4× along sequence, lightning indexer picks top-k compressed blocks, plus a sliding window of uncompressed recent tokens) and **HCA** (compresses 128× and attends densely over the compressed blocks). Most KV stored in **FP8**, BF16 kept only for RoPE dims.
- **Why:** the 1M-context target. Cache reduced to ~2% of standard attention.
- **Impact:** 128k → **1M default context** without proportional cost. Demonstrates the 2026 pattern: not one attention mechanism, but a **portfolio of them at different compression ratios, interleaved by layer.**

### The linear-attention sub-branch

This one has its own lineage, and it re-merged with the trunk in 2025.

**Linear attention (2020, Katharopoulos)** — drop the softmax, use a kernel feature map, exploit associativity: `(φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV)`. Now O(N) and expressible as an **RNN with a matrix-valued state**. Constant memory at decode — *no KV cache at all*. Cost: fixed-size state means finite memory, and early versions were badly worse than softmax.

- **RetNet / RWKV / GLA (2023)** — add **decay/forgetting** to that state. GLA (Gated Linear Attention) makes the decay **data-dependent** and gives it a hardware-efficient chunkwise form.
- **Mamba (2023) → Mamba-2 (2024)** — selective state-space models; Mamba-2's SSD duality showed SSMs and linear attention are the *same family*, which is what let them be dropped into transformer stacks.
- **DeltaNet (Schlag 2021; Yang et al. 2024)** — replace "accumulate into state" with the **delta rule**: write a *correction* (erase the old association for this key, write the new one). Yang et al.'s contribution was making it parallelizable over sequence length, which is what made it trainable at scale.
- **✦ Gated DeltaNet (2024/ICLR 2025, Yang et al.)** — delta rule **plus** a gated decay. Erase-then-write *and* forget-over-time. Empirically the first linear variant genuinely competitive with softmax attention on recall-heavy tasks.
- **✦ KDA — Kimi Delta Attention (Oct 2025, Moonshot)** — Gated DeltaNet with **finer-grained (per-channel) gating**, implemented via a Diagonal-Plus-Low-Rank transition matrix with a custom chunkwise kernel. Better use of a finite-state memory.

### ✦ Hybrid stacks (2025–2026) — the actual resolution

- **Parent:** ← softmax attention ⊕ linear attention
- **Edge:** **stop choosing.** Interleave linear-attention layers with a minority of full-attention layers.
- **Why:** linear layers give O(N) and no cache; the sparse full-attention layers preserve exact long-range recall. You get most of both.
- **Impact:** this is where the frontier actually landed.
  - **Qwen3-Next / Qwen3.5** — Gated DeltaNet : full attention at **3:1**, plus output gating.
  - **Kimi Linear → Kimi K3** — KDA : gated MLA, 20 : 7 layers (**≈3:1**). 75% KV-cache reduction, ~6× decode throughput at 1M context.
  - **Ling 2.5** — 7:1, the most aggressive published ratio.
  - **Nemotron 3** — Mamba-2 + transformer + MoE; ~3.8:1 (Nano) to 5:1 (Super).
  - **MiniMax M2.5** — the dissenter: went **back to plain MHA** for reliability.

> As of 2026, **there is no consensus on attention.** That is the headline. GQA, MLA, MLA+DSA, gated-DeltaNet-hybrid, KDA-hybrid, and plain MHA are all shipping in frontier models simultaneously.

---

<a id="b7"></a>

## Branch 7 — Systems, not math

A branch where the model is **bit-for-bit unchanged** and only the execution changes. Easy to under-rate; it moved the frontier more than several architectural nodes did.

### ✦ FlashAttention (2022) — Dao et al.

- **Parent:** ← standard attention kernels
- **Edge:** never materialize the N×N score matrix in HBM. Tile into SRAM, do scores→softmax→weighted-sum per tile with a streaming (still exact) online softmax, write only the output.
- **Why:** attention was **memory-bandwidth bound**, not FLOP bound. ALUs idled waiting on HBM.
- **Impact:** ✦ memory for attention becomes **linear** in N. This — not any architectural change — is what made 8k/32k/128k context practically trainable. FA2 (2023) fixed work partitioning; FA3 (2024) exploits Hopper async + FP8. [→ `llm_docs`](llm_docs.md#flashattention)

### PagedAttention / vLLM (2023) — Kwon et al.

- **Parent:** ← FlashAttention era serving
- **Edge:** manage the KV cache like **virtual memory** — fixed-size pages, non-contiguous, shared across sequences with copy-on-write.
- **Why:** naive contiguous cache allocation wasted 60–80% of memory to fragmentation and over-reservation.
- **Impact:** 2–4× serving throughput with zero model change. Prefix sharing across requests falls out for free.

### Speculative decoding (2022–23) → Multi-Token Prediction (2024)

- **Parent:** ← autoregressive decode
- **Edge:** decode is memory-bound and wastes the GPU one token at a time. **Speculative decoding**: a small draft model proposes k tokens, the big model verifies them in one forward, accepting the longest correct prefix — provably the same output distribution. **MTP** (Gloeckle et al. 2024; DeepSeek-V3): train the model with *extra heads predicting t+2, t+3…*, so it can draft for itself.
- **Why:** free wall-clock at equal quality.
- **Impact:** MTP is now a *training objective* in DeepSeek V3/R1, Qwen3, GLM, MiniMax M2 — it both improves the base model (denser signal per token) and gives you a built-in drafter.

### Quantization: post-hoc → native

- **Parent:** ← FP16/BF16 training and serving
- **Edge:** progressively fewer bits, progressively earlier in the pipeline.
  - 2022–23, **post-hoc**: LLM.int8, GPTQ, AWQ, QLoRA — quantize a finished model.
  - 2024, **training in FP8**: DeepSeek-V3 trains in FP8 with per-block scaling.
  - 2025, **native low-bit release**: gpt-oss ships MXFP4 MoE weights.
  - 2026, **QAT from SFT onward**: Kimi K3 trains quantization-aware from the SFT stage with **MXFP4 weights / MXFP8 activations**.
- **Why:** every halving of bits halves memory bandwidth, which is the binding constraint at decode.
- **Impact:** the trend line is unambiguous — **low precision is migrating from a deployment afterthought into the training objective itself.**

---

<a id="b8"></a>

## Branch 8 — Conditional compute (MoE)

The other way to get capacity without paying for it: make most of the model idle.

### Mixture of Experts (1991, Jacobs et al.) → ✦ Sparsely-Gated MoE (2017, Shazeer et al.)

- **Parent:** ← ensemble learning
- **Edge:** 2017's version — a gating network routes each token to a **top-k of thousands** of expert FFNs. Only those experts compute.
- **Why:** decouple **total parameters** (knowledge capacity) from **active parameters per token** (FLOP cost).
- **Impact:** the whole branch. Introduced its two permanent problems too: **load balancing** (the router collapses onto favourite experts) and **all-to-all communication** cost. [→ `llm_docs`](llm_docs.md#moe)

### GShard (2020) → Switch Transformer (2021) → ST-MoE (2022) — Google

- **Parent:** ← Shazeer MoE
- **Edge:** GShard puts MoE into the **Transformer** at scale with expert-parallel sharding. Switch simplifies to **top-1** routing with a capacity factor and an auxiliary load-balance loss. ST-MoE adds the **router z-loss** for stability.
- **Why:** top-1 is the cheapest possible routing; the aux losses are what stop it collapsing.
- **Impact:** established the standard MoE layer. Also established the standard MoE complaint: aux losses fight the language-modeling loss.

### Mixtral 8×7B (2023) — Mistral

- **Parent:** ← Switch
- **Edge:** first widely-used **open** sparse MoE: 8 experts, top-2, 47B total / ~13B active.
- **Why:** prove the economics in the open.
- **Impact:** made MoE the default assumption for open frontier models rather than a Google curiosity.

### ✦ DeepSeekMoE (2024) — DeepSeek

- **Parent:** ← Mixtral / Switch
- **Edge:** two changes. **(1) Fine-grained experts** — split each expert into many smaller ones and route to more of them, hugely increasing the number of expressible expert *combinations* at the same active FLOPs. **(2) Shared experts** — a few experts every token always uses, absorbing common knowledge so the routed experts can genuinely specialize.
- **Why:** with few coarse experts, each one is forced to learn general-purpose stuff, defeating the point.
- **Impact:** ✦ this is the modern MoE layer. DeepSeek V2/V3, Qwen3-MoE, GLM, Kimi and most 2025–26 MoE models are DeepSeekMoE-shaped. The parameter ratios kept getting more extreme: DeepSeek-V3 671B/37B-active → Kimi K2 1T/32B → **Kimi K3 2.8T, 16 experts of 896 active**.

### Auxiliary-loss-free load balancing (2024) — DeepSeek-V3

- **Parent:** ← Switch aux-loss balancing
- **Edge:** delete the aux loss. Instead keep a **per-expert bias** added to router scores, nudged up/down between steps based on observed load.
- **Why:** the aux loss is a gradient that actively degrades the LM objective in exchange for balance. A bias is a *non-gradient* control knob — balance without contaminating the loss.
- **Impact:** now standard. A nice example of the general 2024–26 pattern: **replace a loss-term hack with an explicit control mechanism.**

### 2026: LatentMoE and balanced-by-construction training

- **Parent:** ← DeepSeekMoE
- **Edge:** **LatentMoE** (Kimi K3, Nemotron 3 Super) compresses the large expert linear layers through a latent bottleneck — the same low-rank move MLA made for attention, applied to the FFN. Kimi K3 adds **Quantile Balancing** (derive expert allocation from router-score quantiles) and **fully balanced expert-parallel training with static shapes**.
- **Why:** at 2.8T params, expert imbalance is not a quality problem, it is a *scheduling* problem — one straggler shard stalls the whole step.
- **Impact:** MoE routing is becoming **statically shaped**, which makes it compiler- and hardware-friendly. Expect this to continue.

---

<a id="b9"></a>

## Branch 9 — Alignment and post-training

Everything above produces a *base model*: a next-token predictor. This branch turns it into something you can talk to. It is the branch where **the training regime, not the architecture, is the innovation.**

### ✦ RLHF / InstructGPT (2020–2022) — Christiano et al. → Stiennon et al. → Ouyang et al.

- **Parent:** ← GPT-3 base
- **Edge:** three stages — SFT on demonstrations, train a **reward model** on human preference pairs, then **PPO** against that reward with a KL penalty to the SFT policy.
- **Why:** next-token likelihood optimizes for *plausible continuation*, not *helpful answer*. Those diverge badly.
- **Impact:** ✦ a 1.3B InstructGPT was preferred to 175B GPT-3. This is the node that produced ChatGPT and therefore everything that followed commercially. Cost: a whole RM + PPO pipeline, which is fragile and expensive.

### Constitutional AI / RLAIF (2022) — Anthropic

- **Parent:** ← RLHF
- **Edge:** replace most human preference labels with **model-generated** ones, guided by an explicit written set of principles.
- **Why:** human labeling doesn't scale, and the values are implicit and unauditable.
- **Impact:** synthetic preference data is now standard everywhere; the "explicit principles" idea also seeded the modern model-spec/policy-document practice.

### ✦ DPO (2023) — Rafailov et al.

- **Parent:** ← RLHF
- **Edge:** show that the RLHF objective has a **closed-form optimal policy**, so you can skip the reward model and PPO entirely and optimize preference pairs with a **simple classification loss** on the policy itself.
- **Why:** PPO is unstable, memory-hungry (4 models in memory), and hard to reproduce.
- **Impact:** ✦ democratized alignment — anyone with preference pairs could align a model. Spawned a family (IPO, KTO, ORPO, SimPO). Frontier labs largely moved back to online RL for reasoning, but DPO remains the default for style/safety alignment in open models.

---

<a id="b10"></a>

## Branch 10 — The reasoning / RLVR era

### ✦ Chain-of-thought → verifiers → process rewards (2021–2023)

- **Parent:** ← GPT-3 in-context learning
- **Edge:** prompting the model to produce intermediate steps ("let's think step by step") measurably improves accuracy; then **train** on that — verifiers (Cobbe et al. 2021), self-consistency, STaR (bootstrap on self-generated correct traces), process reward models.
- **Why:** a single forward pass has fixed depth. Serialized tokens are how a transformer buys **variable compute per problem**.
- **Impact:** the necessary groundwork. o1 and R1 did not appear from nowhere — this is a four-year lineage.

### ✦ o1 (Sept 2024) — OpenAI

- **Parent:** ← CoT + RL
- **Edge:** train with large-scale RL to produce a long internal reasoning trace before answering; **scale test-time compute** as a first-class axis.
- **Why:** pretraining compute scaling was hitting data limits. Inference compute was an untapped axis.
- **Impact:** ✦ reframed the scaling story from "bigger model" to "more thinking tokens." Method kept private.

### ✦ DeepSeek-R1 / GRPO (Jan 2025) — DeepSeek

- **Parent:** ← o1 ⊕ DeepSeekMath's GRPO (2024)
- **Edge:** **RLVR** — reinforcement learning from *verifiable* rewards. On math/code, you don't need a reward model at all: just check the answer. **GRPO** drops PPO's value network too, sampling a group of completions and using the group's mean reward as the baseline.
- **Why:** RM-based RL is capped by the RM's quality and is hackable. A verifier is not.
- **Impact:** ✦ the defining regime of 2025–26. R1-Zero showed reasoning behavior emerging from **pure RL with no SFT stage**. Every major lab shipped a "thinking" variant within a year — including, by 2026, **Gemma 4 shipping thinking mode in a 2.3B open model.** GRPO itself got a year of refinements (token-level loss normalization, zero-gradient filtering, active sampling, off-policy corrections).

### Where it is going (2026)

- **Edge:** RLVR expanding past math/code into domains with checkable outputs — theorem proving, chemistry, biology, agentic tool use with executable success criteria.
- **Impact:** the current frontier bet is that **verifiable environments, not parameters, are the scarce resource.** Kimi K3 is explicitly positioned as an *agentic* model; the architecture work (1M context, cheap decode) exists to serve long agentic rollouts.

---

<a id="b11"></a>

## Branch 11 — Multimodality and fusion

### Late fusion / adapter (2021–2023) — CLIP → Flamingo → LLaVA

- **Parent:** ← frozen vision encoder + frozen LLM
- **Edge:** run a separate pretrained vision encoder, project its output into the LLM's token space with a small adapter.
- **Why:** cheapest possible path; reuses two existing pretrained artifacts.
- **Impact:** the standard VLM recipe for years. This is where the BERT/encoder branch went to live.

### Early fusion (2024–2025) — Chameleon, Llama 4

- **Parent:** ← late fusion
- **Edge:** tokenize images (and audio) into the **same sequence** as text and train on mixed data **from pretraining**, not as a bolt-on stage.
- **Why:** adapters keep the modalities in separate representational worlds; cross-modal reasoning suffers.
- **Impact:** "natively multimodal" becomes the standard claim.

### ✦ Encoder-free multimodal (2026) — Gemma 4

- **Parent:** ← early fusion
- **Edge:** delete the vision encoder entirely. Gemma 4's 12B ingests **raw 48×48×3 image patches through a 35M-parameter matmul** (replacing a 550M encoder) and **raw 40ms audio chunks** projected straight into embedding space.
- **Why:** if the LLM is going to learn the visual features anyway, a separate encoder is redundant parameters and a redundant training stage.
- **Impact:** the logical endpoint of fusion. It is also, structurally, the same move as NoPE and as aux-loss-free balancing: **delete the special-purpose component and let the main stack learn it.**

---

<a id="b12"></a>

## Branch 12 — Topology: rethinking the residual stream

The newest branch, and the first in years to touch something the 2017 paper decided. Everything from ResNet 2016 onward assumed `x_{l+1} = x_l + F(x_l)`. 2026 asks whether that `+` is right.

### Hyper-Connections (2024) — ByteDance

- **Parent:** ← residual connections
- **Edge:** widen the residual stream into **n parallel streams** with learned connection weights between them and each layer.
- **Why:** a single residual stream forces a tradeoff between gradient flow and representation collapse; extra width lets the network learn its own depth-wise wiring.
- **Impact:** real gains, but it **broke the identity mapping** that makes residuals stable — training instability and memory overhead at scale.

### ✦ mHC — Manifold-Constrained Hyper-Connections (Dec 2025) — DeepSeek

- **Parent:** ← Hyper-Connections
- **Edge:** project HC's connection space onto a **manifold that restores the identity-mapping property**, then optimize the implementation so the memory overhead is tolerable.
- **Why:** keep HC's expressiveness, get residual stability back.
- **Impact:** shipped in DeepSeek V4. Makes learned residual topology practical at trillion scale.

### ✦ Attention Residuals (Mar 2026) — Moonshot / Kimi

- **Parent:** ← residual connections ⇄ mHC
- **Edge:** replace uniform accumulation with **softmax attention over the outputs of all preceding layers** — each layer learns input-dependent weights over its own history.
- **Why:** in Pre-LN, the residual stream grows with depth and each layer's contribution is progressively **diluted**, forcing late layers to emit ever-larger outputs to be heard. (This is the same variance-growth problem `llm_docs` describes for the 1/√N init — AttnRes attacks it structurally instead of at init.)
- **Impact:** **1.25× compute-scaling advantage** (matches a model given 25% more compute), +7.5% GPQA-Diamond, +3.6% MATH, +3.1% HumanEval, for <4% training and <2% inference overhead. *Block AttnRes* (~8 blocks) captures most of the gain cheaply. Shipped in Kimi K3.

> Two labs, two independent answers, same year, same question: **"the residual stream is a fixed unweighted sum — why?"** This is the most interesting live branch in the graph. [→ A.2](#a2)

---

<a id="b13"></a>

## Branch 13 — Optimizers and numerics

Usually invisible, occasionally decisive.

### Adam → AdamW (2017) → muP (2022)

- **Parent:** ← SGD
- **Edge:** AdamW decouples weight decay from the adaptive step. **muP** (Maximal Update Parametrization) reparametrizes initialization and learning rates so that **optimal hyperparameters transfer across width** — tune on a small model, apply to the large one.
- **Why:** a hyperparameter sweep at 100B+ is unaffordable. muP makes it a small-model problem.
- **Impact:** standard practice at frontier labs; the reason large runs mostly work on the first try now.

### ✦ Muon (2024) → MuonClip (2025) → per-head Muon (2026)

- **Parent:** ← AdamW
- **Edge:** **Muon** treats 2D weight matrices as matrices, not flat vectors — it orthogonalizes the momentum update (Newton–Schulz iteration) so the update has balanced singular values. Roughly ~2× token efficiency vs AdamW in reported comparisons. **MuonClip** (Kimi K2) adds **QK-Clip**: after each step, rescale the Wq/Wk weights whenever max attention logits exceed a threshold. **Per-head Muon** (Kimi K3) applies the orthogonalization per attention head.
- **Why:** Muon at trillion scale produced exploding attention logits and loss spikes. QK-Clip attacks the cause at the weights rather than damping the symptom.
- **Impact:** Kimi K2 reported **zero loss spikes across 15.5T tokens** — the practical proof that a non-Adam optimizer works at frontier scale. This is the first serious challenge to Adam's 10-year monopoly. Note the pairing with QK-Norm ([B4](#b4)): the *same* logit-explosion problem, solved architecturally by one camp and in the optimizer by the other.

---

<a id="convergence"></a>

## Where the frontier stands (2026)

What today's models actually ship. Read the columns — the disagreements are the interesting part.

| | **DeepSeek V4** | **Kimi K3** | **Qwen3.5** | **Gemma 4** | **GLM-5** | **MiniMax M2.5** |
|---|---|---|---|---|---|---|
| Total / active | 1.6T / 49B | 2.8T / 16-of-896 exp | 397B / 17B | 2.3B–31B | — | — / 10B |
| Attention | CSA + HCA (sparse, interleaved) | KDA : gated MLA **20:7** | Gated DeltaNet : full **3:1** | GQA + SWA **5:1** | MLA + DSA | **plain MHA** |
| Position | RoPE | **NoPE** | RoPE | pp-RoPE 1M/10k split | RoPE | RoPE |
| FFN | MoE | Stable **LatentMoE** | MoE | dense (MoE at 26B-A4B) | MoE | MoE |
| Residual topology | **mHC** | **AttnRes** | standard | standard | standard | standard |
| Norm | RMSNorm | RMSNorm + SiTU | RMSNorm + QK-Norm | RMSNorm pre+post, QKNorm | RMSNorm | RMSNorm + QK-Norm |
| Numerics | KV in FP8 | **MXFP4 w / MXFP8 a, QAT from SFT** | — | — | — | — |
| Context | **1M** | **1M** | 262K | 32K std / 128–256K long-ctx evals | — | — |
| Post-training | RLVR | RLVR, agentic | RLVR | RLVR + distillation, thinking mode | RLVR | RLVR |

**What *is* settled (universal in 2026):**

- Decoder-only, Pre-LN, RMSNorm, SwiGLU-family gated FFN
- Byte-level BPE
- Sparse MoE for anything above ~30B
- Overtraining far past Chinchilla; distillation for small models
- RLVR post-training and an explicit thinking/reasoning mode
- FlashAttention-family kernels, paged KV serving

**What is emphatically *not* settled:**

- **Attention.** Six live answers. No convergence.
- **Position.** RoPE vs NoPE vs per-layer mixtures.
- **Residual topology.** Standard vs mHC vs AttnRes — brand new, two independent answers.
- **Optimizer.** AdamW vs the Muon family.
- **Precision.** Post-hoc quantization vs QAT-from-SFT.

---

<a id="open"></a>

## Open branches — unresolved bets

Reading the graph forward, these are the edges being drawn right now:

1. **Does linear attention actually win, or is the hybrid ratio a permanent compromise?** Nobody has shipped 0% full attention. The ratios cluster hard — 3:1 for Qwen3.5 and Kimi, 3.8–5:1 for Nemotron, 7:1 for Ling — which usually means a local optimum nobody has escaped, not a law.
2. **Is NoPE viable model-wide?** Kimi K3 bets yes, leaning on recurrent linear layers to carry position. If it holds at 2.8T, RoPE becomes optional after 5 years of universality.
3. **How far does topology go?** mHC and AttnRes both re-weight depth. The obvious next question is whether *depth itself* becomes dynamic (per-token layer routing / early exit) — an old idea that never worked, now with much better plumbing.
4. **Does MoE sparsity keep increasing?** 8-of-64 → 8-of-256 → 16-of-896. The ratio of total to active params keeps climbing. There is presumably a floor where the router stops having enough signal.
5. **Does the closed frontier look like this at all?** GPT-5.x, Gemini 3.x, and Claude publish no architecture. Every node in this graph after 2023 is inferred from **open** models. It is entirely possible the closed labs made different bets, and we would not know.
6. **What replaces verifiable rewards?** RLVR works where answers are checkable. The scaling story past math/code/agentic-tools is genuinely unsolved.

---

<a id="appendix"></a>

## Appendix — deep dives

<a id="a1"></a>

### A.1 — Why MLA beats GQA, mechanically

Both shrink the KV cache. They differ in *what they destroy*.

**GQA destroys diversity.** With g groups, heads inside a group see **identical** keys and values. The only thing distinguishing them is their query projection. From `llm_docs`' framing of multi-head attention — heads exist so that different subspaces can be attended at different positions — GQA collapses the *key/value* subspaces while keeping the query ones. At `g=1` (MQA), all 32 heads attend over the same K/V, and the quality drop is measurable.

**MLA destroys rank.** Cache `c^KV = W_DKV · x`, a low-dimensional latent (e.g. 512 vs 32 heads × 128 = 4096). At use time, `K_i = W_UK_i · c^KV`, `V_i = W_UV_i · c^KV` — every head gets **its own distinct** K and V. They all live in a shared low-rank subspace, but they are not identical.

The trade: GQA says "few distinct K/V vectors, each full-rank." MLA says "many distinct K/V vectors, all drawn from a low-rank subspace." Empirically the second is the better bargain — head *diversity* matters more than head *rank*.

Two implementation notes that make it practical:

- **The up-projections fold into the surrounding weights.** `W_UK` can be absorbed into `W_Q` (since `q·K = q·(W_UK·c) = (W_UK^T·q)·c`), so at inference you never materialize full K — you attend directly against the latent. This is why MLA is not just a memory win but often a compute win too.
- **RoPE breaks that folding.** Rotation is position-dependent, so it cannot commute through a fixed matrix. DeepSeek's fix is **decoupled RoPE**: carry a small extra set of dimensions (e.g. 64) that *do* get rotated and *are* cached separately, alongside the un-rotated latent. That awkwardness is a real cost of MLA and is part of why Kimi K3 dropping RoPE for NoPE is interesting — it removes the exception.

<a id="a2"></a>

### A.2 — The residual-stream dilution problem

`llm_docs` already sets this up in the GPT-2 section, from the *initialization* angle: in Pre-LN, `LN(x_l)` has unit variance, so every sublayer output has roughly constant variance `c`, added unnormalized into the stream. Therefore `Var(x_l) ≈ Var(x_0) + (l−1)·c` — **linear in depth** — and the fix was scaling output projections by `1/√N`.

That fix is an **initialization-time** fix. It makes the *total* O(1) at step 0. It does not constrain what happens after training starts, and it does not address the *relative* problem: by layer 80, the stream is a sum of 80 contributions, so any single layer's output is ~1/80th of what the next layer reads. To have influence, late layers must learn to emit **large** outputs. That is the dilution AttnRes names.

Three responses, in order of how much they change:

1. **1/√N init (GPT-2, 2019)** — control the scale at init. Cheap, universal, insufficient at 100+ layers.
2. **Hyper-Connections → mHC (2024–25)** — widen the stream to n copies and *learn* the connection weights, then constrain them to a manifold so the identity path survives. Learned but **static** wiring — the weights don't depend on the input.
3. **AttnRes (2026)** — make the weights **input-dependent**: softmax attention over previous layer outputs. The learned patterns come out diagonally dominant (each layer mostly reads its predecessor, i.e. it *rediscovers* the standard residual) plus sparse long skips to specific earlier layers. That is a nice result — it means the standard residual connection was approximately right, and the gain comes from the exceptions.

Cost accounting matters here, because this is why it ships: full AttnRes needs every previous layer's output live, which is a memory problem under pipeline parallelism. **Block AttnRes** groups layers into ~8 blocks and attends over block outputs — 5.5× the per-layer dimension of activations vs 3× for standard residuals, <4% training wall-clock, <2% inference latency, for a 1.25× compute-efficiency gain. That is an unusually good ratio, which is why two labs converged on the area at once.

<a id="a3"></a>

### A.3 — Why "efficiency" edges dominate after 2023

A structural observation worth stating explicitly, because it explains the shape of the graph.

Before GPT-3, the deliverable was a *fine-tuned model* and the cost that mattered was **training** cost. Post-GPT-3, the deliverable is a *served endpoint*, and a frontier model is trained once and run billions of times. Inference cost dominates total cost of ownership by orders of magnitude.

Every branch below the LLaMA node is therefore attacking one of exactly three inference bottlenecks:

| Bottleneck | Branch | Representative edges |
|---|---|---|
| **Memory bandwidth** (decode is bandwidth-bound, not FLOP-bound) | 6, 7 | MQA/GQA/MLA, linear attention, FP8/MXFP4 KV, quantization |
| **FLOPs per token** | 8, 6 | MoE (fewer active params), sparse attention (fewer scored pairs) |
| **Wall-clock serialization** (one token at a time) | 7 | speculative decoding, MTP, PagedAttention batching |

And the two branches that *aren't* efficiency — RLVR (B10) and topology (B12) — are both about buying quality **per unit of compute**: RLVR by spending inference compute deliberately, topology by making each parameter's contribution count for more.

There is no branch after 2023 whose pitch is "this is bigger." That era ended with Chinchilla.

<a id="a4"></a>

### A.4 — Recurring meta-patterns

Once you have the whole graph in front of you, the same five moves keep recurring. Useful for predicting the next edge.

1. **Turn a binary into a dial.** MQA vs MHA → GQA's `g`. Dense vs MoE → active/total ratio. Full vs linear attention → hybrid layer ratio. When two options each sacrifice something, someone will parametrize the space between them, and that parametrization usually wins.
2. **Replace a loss-term hack with an explicit control mechanism.** MoE aux load-balance loss → per-expert bias. Attention logit explosion → QK-Norm / QK-Clip. Gradient penalties → constraints. Losses that fight the main objective get replaced by machinery that doesn't.
3. **Delete the special-purpose component; let the stack learn it.** Encoder + decoder → decoder only. Vision encoder → raw patches (Gemma 4). Positional embeddings → NoPE. Reward model → verifier (RLVR). Draft model → the model's own MTP heads.
4. **Low-rank the biggest tensor you have.** LoRA on weights → MLA on the KV cache → LatentMoE on expert FFNs. Same move, three different tensors, three different years.
5. **Move the optimization earlier in the pipeline.** Quantization: post-hoc → training in FP8 → QAT from SFT. Sparsity: post-hoc pruning → natively-trained sparse attention (NSA). Anything that starts as a deployment trick eventually becomes part of training.

If you want to guess what 2027 looks like, apply pattern 3 to whatever component still looks bolted-on, and pattern 5 to whatever is still done after training.

---

<a id="sources"></a>

## Sources

**Reference file:** `llm_docs.md` (Transformer, GPT-1/2, RoPE, RMSNorm, SwiGLU, MQA/GQA, FlashAttention, MLA, MoE).

**Papers, primary:**

- Vaswani et al. 2017, *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau et al. 2014 — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Radford et al. 2018 (GPT-1); Radford et al. 2019 (GPT-2); Brown et al. 2020 (GPT-3) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Devlin et al. 2018, BERT — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Kaplan et al. 2020 — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361); Hoffmann et al. 2022, Chinchilla — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Su et al. 2021, RoFormer/RoPE — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Zhang & Sennrich 2019, RMSNorm — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Shazeer 2020, GLU Variants — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Touvron et al. 2023, LLaMA — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Shazeer 2019, MQA — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150); Ainslie et al. 2023, GQA — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- DeepSeek-V2 2024 (MLA + DeepSeekMoE) — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434); DeepSeek-V3 — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek NSA 2025 — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089); DeepSeek-V3.2 (DSA) — [arXiv:2512.02556](https://arxiv.org/pdf/2512.02556)
- Dao et al. 2022, FlashAttention — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Shazeer et al. 2017, Sparsely-Gated MoE — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538); Fedus et al. 2021, Switch — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Katharopoulos et al. 2020, Linear Transformers — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Yang et al. 2024, *Parallelizing Linear Transformers with the Delta Rule* — [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- Yang et al. 2024, *Gated Delta Networks* — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Gu & Dao 2023, Mamba — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Ouyang et al. 2022, InstructGPT — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155); Rafailov et al. 2023, DPO — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- DeepSeek-R1 2025 — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) · [Nature version](https://www.nature.com/articles/s41586-025-09422-z)
- Kimi K2 2025 — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- Kimi Linear / KDA 2025 — [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- mHC: Manifold-Constrained Hyper-Connections 2025 — [huggingface.co/papers/2512.24880](https://huggingface.co/papers/2512.24880)
- Attention Residuals 2026 (Kimi Team) — [huggingface.co/papers/2603.15031](https://huggingface.co/papers/2603.15031)
- Gemma 4 Technical Report 2026 — [arXiv:2607.02770](https://arxiv.org/abs/2607.02770)
- Gated DeltaNet-2 2026 — [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)

**Secondary / model documentation:**

- [Kimi K3 tech blog](https://www.kimi.ai/blog/kimi-k3) · [Raschka, Kimi K3 architecture notes](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html)
- [Raschka, LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) · [DeepSeek Sparse Attention entry](https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/)
- [Raschka, The State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [Labonne, *Qwen3.5: Nobody Agrees on Attention Anymore*](https://huggingface.co/blog/mlabonne/qwen35)
- [DeepSeek V4 architecture overview](https://www.morphllm.com/deepseek-v4)
- [Meta, The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [vLLM blog, DeepSeek-V3.2 sparse attention](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)

---

*[↑ Index](#index)*
