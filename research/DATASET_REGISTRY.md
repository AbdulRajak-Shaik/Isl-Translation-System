# Verified Dataset Registry: Indian Sign Language (ISL) Resources

This registry documents all audited publicly available datasets, benchmarks, dictionaries, and parallel corpora for Indian Sign Language. Every entry separates verified empirical facts from assumptions.

---

## 1. Primary Continuous Translation Datasets

### 1.1 ISLTranslate
* **Dataset Name:** ISLTranslate
* **Official Source URL:** [https://github.com/Exploration-Lab/ISLTranslate](https://github.com/Exploration-Lab/ISLTranslate)
* **Research Paper:** *"ISLTranslate: Dataset for Translating Indian Sign Language"*, Findings of ACL 2023 ([ACL Anthology: 2023.findings-acl.665](https://aclanthology.org/2023.findings-acl.665/))
* **Authors:** Abhinav Joshi, Susmit Agrawal, Ashutosh Modi (Exploration-Lab, IIT Kanpur & IIT Hyderabad)
* **Verified Volume:**
  - 31,222 continuous video-to-sentence/phrase pairs
  - 11,655 unique vocabulary items
* **Modality / Task Type:** Continuous Sign Language Translation (CSLT) — Gloss-Free End-to-End Pose-to-Text and Video-to-Text
* **Languages Available:** ISL (Visual video & extracted skeletal pose) → English text
* **Annotation Format:** Aligned sentence-level English translations; frame-level bounding boxes and pose keypoints
* **Signer Diversity:** Multiple adult deaf signers and certified ISL instructors recorded in structured indoor studios
* **Resolution & Duration:** 720p / 1080p RGB, 25/30 FPS; clip durations typically 2 to 12 seconds per sentence
* **License & Restrictions:** Research and Non-Commercial Educational use only (Academic License via Exploration-Lab)
* **Download & Access:** GitHub repository provides metadata, pose features, and download scripts for authorized research
* **Overlap:** Serves as the core continuous foundation for the subsequent larger iSign benchmark
* **Recommended Project Role:** **Primary continuous translation benchmark and evaluation target.**

---

### 1.2 iSign (ACL 2024 Benchmark)
* **Dataset Name:** iSign: A Benchmark for Indian Sign Language Processing
* **Official Source URL:** [https://exploration-lab.github.io/iSign/](https://exploration-lab.github.io/iSign/)
* **Research Paper:** *"iSign: Benchmark for Indian Sign Language"*, Findings of ACL 2024 ([ACL Anthology: 2024.findings-acl.643](https://aclanthology.org/2024.findings-acl.643/))
* **Authors:** Abhinav Joshi, et al. (Exploration-Lab, IIT Kanpur, Max Planck Institute, ISLRTC, Microsoft IDC)
* **Verified Volume:**
  - 118,000+ continuous video-sentence/phrase pairs
  - Tasks: SignVideo2Text, SignPose2Text, Text2Pose, Word Prediction, Sign Semantics
* **Modality / Task Type:** Continuous Video-to-Text, 3D Pose-to-Text, Sign Language Production
* **Languages Available:** ISL → English spoken translations
* **Annotation Format:** Normalized text sentences, aligned timestamps, OpenPose/MediaPipe Holistic 3D skeletal landmarks
* **Signer Diversity:** Broad diversity across multiple certified ISLRTC instructors, ISH News deaf presenters, and community contributors
* **Resolution & Duration:** 1080p/720p, 30 FPS; 3 to 25 seconds per utterance
* **License & Restrictions:** Research-Only Agreement. Commercial exploitation strictly prohibited without express institutional permissions
* **Download & Access:** Controlled access via Hugging Face (`Exploration-Lab/iSign`) and academic verification form
* **Overlap:** Incorporates and expands ISLTranslate and ISH News corpuses
* **Recommended Project Role:** **Long-term benchmark evaluation target and large-scale pretraining reference.**

---

### 1.3 ISL-CSLTR
* **Dataset Name:** ISL-CSLTR (Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition)
* **Official Source URL:** [https://data.mendeley.com/datasets/kcmpdxky7p/1](https://data.mendeley.com/datasets/kcmpdxky7p/1)
* **DOI:** `10.17632/kcmpdxky7p.1` (Published: January 2021)
* **Authors:** Elakkiya R. and Natarajan B.
* **Verified Volume:**
  - 700 continuous videos
  - 18,863 sentence-level frames
  - 1,036 word-level images
  - 100 distinct spoken sentences
* **Modality / Task Type:** Continuous Sign Language Recognition and Translation
* **Languages Available:** ISL → English text & gloss tokens
* **Annotation Format:** Plain text transcripts, sentence-level boundary timestamps, word mapping CSVs
* **Signer Diversity:** 7 distinct signers (including native deaf students and hearing volunteers)
* **Resolution & Duration:** 1280×720 at 30 FPS; ~3 to 8 seconds per sentence
* **License & Restrictions:** CC BY 4.0 (Open access for academic and commercial use with attribution)
* **Download & Access:** Direct public download from Mendeley Data repository
* **Overlap:** Independent dataset; zero overlap with Exploration-Lab corpora
* **Recommended Project Role:** **Primary controlled local smoke-test and unit-testing corpus due to open CC BY 4.0 licensing and compact footprint.**

---

## 2. Large-Scale Isolated Sign Datasets (Auxiliary Representation Learning)

### 2.1 CISLR (Corpus for Indian Sign Language Recognition)
* **Dataset Name:** CISLR
* **Official Source URL:** [https://github.com/Exploration-Lab/CISLR](https://github.com/Exploration-Lab/CISLR) / [Hugging Face](https://huggingface.co/datasets/Exploration-Lab/CISLR)
* **Research Paper:** Exploration-Lab (EMNLP)
* **Verified Volume:**
  - 4,700+ to 4,765 isolated ISL vocabulary signs
  - Tens of thousands of isolated video instances
* **Modality / Task Type:** Isolated Sign Language Recognition (ISLR) / Word-level classification
* **Languages Available:** ISL → English gloss / word labels
* **Annotation Format:** Word IDs, sign category labels, pre-extracted Inception3D / I3D and 2D/3D pose features
* **Signer Diversity:** Varied native signers in frontal camera poses
* **License & Restrictions:** Academic / Research use only
* **Download & Access:** Hugging Face Hub / GitHub release scripts
* **Overlap:** Shares lexical definitions with ISLRTC dictionary
* **Recommended Project Role:** **Auxiliary representation pretraining for visual encoder and CTC sign vocabulary recognition.**

---

### 2.2 INCLUDE / INCLUDE-50
* **Dataset Name:** INCLUDE (Indian Lexicon Sign Language Dataset)
* **Official Source URL:** [https://zenodo.org/record/4010759](https://zenodo.org/record/4010759)
* **Research Paper:** *"INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition"*, ACM Multimedia 2020 (ACM MM '20, DOI: `10.1145/3394171.3413528`)
* **Authors:** Prem Sridhar, et al.
* **Verified Volume:**
  - 4,287 video clips
  - 263 isolated word signs across 15 semantic categories
  - INCLUDE-50 subset (50 signs) for rapid benchmarking
* **Modality / Task Type:** Isolated Sign Recognition
* **Languages Available:** ISL → English labels
* **Annotation Format:** Per-video sign label CSV, category metadata
* **Signer Diversity:** 7 deaf students and trained signers; varied indoor and outdoor lighting
* **Resolution & Duration:** 1920×1080 Full HD at 30 FPS; ~2 to 4 seconds per clip
* **License & Restrictions:** Research Use Only (Subject to Zenodo Data Agreement)
* **Download & Access:** Hosted on Zenodo; academic credential verification required for video archives
* **Overlap:** None (independent recording sessions)
* **Recommended Project Role:** **Validation of isolated handshape robustness; auxiliary sign-level spatial encoder pretraining.**

---

## 3. Broadcast and Natural Continuous Video Sources

### 3.1 ISH-NEWS Broadcast Corpus
* **Dataset / Source:** Indian Sign Hearing (ISH) News Continuous Translation Corpus
* **Research Publication:** Nature Scientific Reports (DOI: `10.1038/s41598-026-60893-0`)
* **Modality / Task Type:** Continuous Sign Language Broadcast News Translation
* **Verified Facts & Rights Analysis:**
  - Source videos originate from public YouTube broadcasts of ISH News deaf presenters covering daily news in ISL.
  - While YouTube videos are publicly accessible to viewers, **broadcasting rights belong to India Signing Hands Pvt. Ltd.**
  - **Legal Assessment:** Training directly on scraped raw video is restricted under copyright law without an explicit licensing agreement.
  - **Permissible Usage:** Derived, anonymized skeletal landmark coordinates (pose meshes) extracted without redistributing original copyrighted video streams fall under non-consumptive computational research exceptions in specific jurisdictions, but video redistributions are prohibited.
* **Recommended Project Role:** **Benchmark reference only; raw video downloads avoided in local repository.**

---

## 4. Lexical Dictionaries & Semantic Grounding

### 4.1 ISLRTC Official Dictionary (Department of Empowerment of Persons with Disabilities, Govt of India)
* **Official URL:** [https://islrtc.nic.in/](https://islrtc.nic.in/) and [https://data.gov.in/catalog/indian-sign-language-dictionary](https://data.gov.in/catalog/indian-sign-language-dictionary)
* **Authority:** Indian Sign Language Research and Training Centre (ISLRTC), Ministry of Social Justice and Empowerment
* **Verified Volume:**
  - 10,000 standardized ISL lexical signs across 6 editions (Edition 1: 3,000 signs, Edition 2: 6,000 signs, Edition 3+: 10,000 signs)
  - Categories: Educational, legal, medical, technical, administrative, and daily conversational terms
* **Modality / Task Type:** Lexical video dictionary with English & Hindi definitions
* **Languages Available:** ISL → English and Hindi text descriptions
* **License:** Government Open Data License (GODL-India) / Public informational access
* **Recommended Project Role:** **Ground-truth lexical reference, sign disambiguation, and canonical token validation.**

---

## 5. Summary Matrix for Project Implementation

| Dataset | Modality | Volume | License | Primary Role in System |
|---|---|---|---|---|
| **ISLTranslate** | Continuous Sentence | 31,222 pairs | Academic (Exploration-Lab) | Continuous Translation Baseline & Evaluation |
| **iSign** | Continuous Multimodal | 118,000+ pairs | Research-Only | Large-Scale Pretraining & Benchmark Standard |
| **ISL-CSLTR** | Continuous Sentence | 700 videos | CC BY 4.0 | Local Smoke-Tests, Unit Tests, Pipeline Validation |
| **CISLR** | Isolated Word | 4,700 signs | Academic | Auxiliary Sign/CTC Token Representation |
| **INCLUDE** | Isolated Word | 4,287 videos | Academic (Zenodo) | Hand Landmark Robustness Evaluation |
| **ISLRTC Dict** | Lexicon Dictionary | 10,000 signs | GODL-India | Canonical Vocabulary Grounding & Evaluation |
