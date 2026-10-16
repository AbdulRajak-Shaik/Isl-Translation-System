# Formal Evaluation Protocol: Metrics, Splits, and Benchmarking Standards

**System:** Continuous Indian Sign Language Translation to English & Telugu  
**Document Status:** Standard Operating Procedure (SOP)  

---

## 1. Principles of Rigorous Evaluation

In accordance with strict research integrity standards:
1. **Never Report Training Loss as Evaluation:** All reported numbers must strictly derive from unseen validation or held-out test splits.
2. **Leakage Prevention:**
   - **Signer-Disjoint Splits:** Signers appearing in the test set must *never* appear in the training or validation sets.
   - **Source-Disjoint Splits:** Continuous broadcast recordings or recording sessions must not be chopped across train and test sets.
3. **No Metric Cherry-Picking:** Always report standardized SacreBLEU, chrF++, and WER alongside human qualitative ratings.

---

## 2. Quantitative Automated Metrics

### 2.1 BLEU (Bilingual Evaluation Understudy)
- We evaluate using standard **SacreBLEU** with default tokenization:
  - **BLEU-1, BLEU-2, BLEU-3, BLEU-4**
  - Measures n-gram precision against ground-truth English reference translations.
  - Baseline Target on ISLTranslate / iSign: BLEU-4 $\ge 18.0$.

### 2.2 chrF++ (Character n-gram F-score)
- Character-level F-score is critical for morphologically rich translation (and especially downstream Telugu text):
  - Invariant to minor tokenization differences or inflectional suffixes.
  - Formula:
    $$\text{chrF} = (1 + \beta^2) \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \text{chrP} + \text{chrR}}$$
  - Baseline Target: chrF++ $\ge 42.0$.

### 2.3 Word Error Rate (WER) & Character Error Rate (CER)
- For the auxiliary CTC recognition head and gloss sequences:
  $$\text{WER} = \frac{S + D + I}{N}$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference tokens.

### 2.4 Semantic Critical Tests
- **Negation Inversion Rate:** Frequency with which a negative sign is erroneously translated as affirmative (e.g., *"I did not go"* translated as *"I went"*). Target: $< 3\%$.
- **Question Preservation Rate:** Whether interrogative syntactic mood (who, what, where, why, or yes/no question mark) is preserved. Target: $> 90\%$.
- **Number & Proper Noun Accuracy:** Exact-match precision on signed numerical quantities and names.

---

## 3. Human Evaluation Protocol

Automated metrics alone do not capture whether a deaf signer's true intent was conveyed. A formal human evaluation protocol is instituted:

| Criterion | Scale | Description |
|---|:---:|---|
| **Meaning Preservation (Adequacy)** | 1 – 5 | 5: Complete semantic equivalence. 1: Completely incorrect meaning. |
| **English Fluency** | 1 – 5 | 5: Natural, idiomatic spoken English. 1: Incomprehensible word salad. |
| **Telugu Script Fluency** | 1 – 5 | 5: Natural native Telugu phrasing. 1: Broken or garbled Telugu script. |
| **Temporal Alignment** | 1 – 5 | 5: Live subtitles appear synchronously with signing actions. 1: Severe lag or premature firing. |

*Human evaluation panels should comprise certified ISL interpreters and native deaf signers.*

---

## 4. Inference Latency & Computational Benchmarking

Every released model checkpoint must be benchmarked under standard operational conditions:
- **Streaming Window Latency:** Time to ingest a 60-frame buffer, run holistic extraction, forward through encoder-decoder, and return English + Telugu translation. Target: $< 150\text{ ms}$ on GPU, $< 350\text{ ms}$ on modern multi-core CPU.
- **Peak Memory Usage:** Target $< 1.5\text{ GB}$ VRAM / RAM during real-time streaming.
- **Throughput:** Supported simultaneous video streams per worker instance.
