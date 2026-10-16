# Continuous Translation Error Analysis & Diagnostic Categorization

**System:** Continuous Indian Sign Language Translation  
**Methodology:** Taxonomy of Translation and Landmark Extraction Errors  

---

## 1. Error Taxonomy

When evaluating continuous sign language translation models, errors are categorized into four distinct linguistic and computer vision strata:

```
                              ┌──────────────────────────────────┐
                              │     ISL ERROR CLASSIFICATION     │
                              └────────────────┬─────────────────┘
                                               │
         ┌──────────────────┬──────────────────┴──────────────────┬──────────────────┐
         ▼                  ▼                                     ▼                  ▼
┌──────────────────┐ ┌──────────────────┐               ┌──────────────────┐ ┌──────────────────┐
│ Spatial/Pose     │ │ Boundary &       │               │ Syntactic &      │ │ MT Cascade       │
│ Tracking Errors  │ │ Segmentation     │               │ Grammar Errors   │ │ Errors           │
└──────────────────┘ └──────────────────┘               └──────────────────┘ └──────────────────┘
```

---

## 2. Category Details and Mitigation Strategies

### 2.1 Spatial Tracking & Landmark Dropouts
* **Symptom:** Fast transitions cause hand landmark points to jitter or collapse toward the wrist.
* **Impact:** Loss of handshape nuance (e.g., distinguishing between a closed fist and an index point).
* **Mitigation:**
  - Robust temporal interpolation across missing frames.
  - Feeding both position coordinates $(x, y, z)$ and velocity deltas $(\Delta x, \Delta y, \Delta z)$ into the encoder input projection.

### 2.2 Boundary Detection and Sentence Segmentation
* **Symptom:** In continuous signing, signers do not hold static poses between sentences. Models can prematurely emit a sentence or merge two clauses into a single run-on sentence.
* **Impact:** Distorted sentence semantics and translation latency.
* **Mitigation:**
  - Dynamic sliding window with 50% overlap.
  - Rest-pose detection: Monitoring hand height relative to hip level to identify true discourse boundaries.

### 2.3 Syntactic Inversion & Negation Dropping
* **Symptom:** ISL encodes negation often via headshakes and facial markers rather than an overt lexical "NOT" token. If the facial landmarks are weak, the model drops negation and outputs an affirmative sentence.
* **Impact:** Catastrophic failure in medical or safety contexts.
* **Mitigation:**
  - Multi-scale loss penalty: Triple loss weight on negation tokens during training.
  - Dedicated facial feature slice extracted from eye and brow keypoints.

### 2.4 Telugu MT Cascade Distortions
* **Symptom:** English grammatical ambiguity (such as missing pronouns or ambiguous gender) translated incorrectly into Telugu inflections.
* **Impact:** Inappropriate Telugu verb conjugation (e.g., gender honorific mismatch).
* **Mitigation:**
  - Using IndicTrans2 context-aware prompting.
  - Flagging low-confidence English outputs prior to Telugu translation.
