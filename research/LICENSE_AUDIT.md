# Legal and Licensing Audit: Data, Models, and Software

**Project:** Professional Indian Sign Language Recognition & Translation System  
**Audit Date:** October 2026  
**Auditor:** Software Architect & Legal Compliance Lead  

---

## 1. Executive Licensing Principles

1. **Strict Respect for Academic & Non-Commercial Terms:** Datasets marked "Research-Only" (such as ISLTranslate, iSign, and INCLUDE) must never be embedded in proprietary commercial products without negotiated commercial licenses.
2. **No Video Scraping / Redistribution:** Original high-resolution video streams of deaf individuals are subject to privacy rights and copyright. Under no circumstances will raw third-party videos be bundled into public Git repositories.
3. **Derived Feature Exemption:** In multiple computational jurisdictions, extracting non-reconstructible mathematical vector features (such as 3D skeletal landmark coordinates) from lawfully accessed data for academic analysis does not constitute copyright infringement or video redistribution.
4. **Data Protection & Biometrics (DPDP Act 2023 Compliance):** Under the Indian Digital Personal Data Protection Act (DPDP Act 2023), facial and bodily recordings constitute personal identifiable information (PII). Video processing pipelines must provide local client-side processing options and ensure that server-uploaded frames are discarded from memory immediately following inference.

---

## 2. Dataset-by-Dataset Audit

| Asset | Primary License | Commercial Use Permitted? | Redistribution of Raw Media? | Specific Conditions & Compliance Actions |
|---|---|:---:|:---:|---|
| **ISLTranslate** | Academic Research License (Exploration-Lab) | ❌ NO | ❌ NO | Must cite ACL 2023 paper. Restrict access to registered academic environments. Features cached in `.gitignored` folders. |
| **iSign** | Exploration-Lab Academic Agreement | ❌ NO | ❌ NO | Must cite ACL 2024 paper. Only extracted pose landmarks and normalized text annotations used in model training. |
| **ISL-CSLTR** | Creative Commons Attribution 4.0 (CC BY 4.0) | ✅ YES (with attribution) | ✅ YES (with attribution) | Authors: Elakkiya R. & Natarajan B. (Mendeley Data). Permitted for open experiments, unit test fixtures, and validation. |
| **CISLR** | Academic / Educational Use | ❌ NO | ❌ NO | Word-level feature weights used strictly for auxiliary representation research. |
| **INCLUDE** | Zenodo Data Agreement / Non-Commercial | ❌ NO | ❌ NO | ACM MM 2020 citation required. Download requires authorized Zenodo access credentials. |
| **ISH News Broadcasts** | Proprietary Copyright (India Signing Hands Pvt. Ltd.) | ❌ STRICTLY NO | ❌ STRICTLY NO | Raw video scraping prohibited. Video rights reside exclusively with broadcaster. |
| **ISLRTC Dictionary** | Government Open Data License (GODL-India) | ✅ YES (informational public use) | ✅ YES (with attribution) | Hosted on `data.gov.in`. Official definitions used as vocabulary ground-truth. |

---

## 3. Translation and Speech Generation Component Licensing

### 3.1 AI4Bharat IndicTrans2
* **Repository:** [https://github.com/AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
* **License:** MIT License / Creative Commons Attribution-NonCommercial (CC-BY-NC) for specific large pre-trained weights.
* **Compliance Action:**
  - The model weights and `IndicTransToolkit` code are open source for academic and non-commercial research.
  - Checkpoint versions (such as `ai4bharat/indictrans2-en-indic-dist-200M`) are explicitly supported in the modular backend translation service.
  - A fallback translation interface is implemented so that commercial deployers can substitute an enterprise translation endpoint if required.

### 3.2 Speech Synthesis (TTS)
* **gTTS (Google Text-to-Speech API client):**
  - Uses the public Google Translate speech synthesis endpoint.
  - License: MIT License (client code).
  - Production Note: Subject to Google TOS and rate limits; suitable for development and research verification.
* **W3C Web Speech API (Client-side):**
  - Native browser implementation running on the user's OS.
  - License: Standard Web API with zero external licensing fees or data transmission.
* **Microsoft Edge TTS (`edge-tts`):**
  - Free neural speech engine supporting Indian English (`en-IN-NeerjaNeural`, `en-IN-PrabhatNeural`) and Telugu (`te-IN-ShrutiNeural`, `te-IN-MohanNeural`).
  - License: Open-source wrapper (GNU LGPL / Python community).

---

## 4. Privacy and Video Data Retention Policy

To adhere to ethical computer vision standards:
1. **Live Webcam Streaming:** Video frames streamed over WebSockets are processed purely in ephemeral RAM buffers. No video frames or audio buffers are written to persistent disk storage during live inference.
2. **Uploaded Video Files:** Uploaded video files are saved to a temporary sandboxed directory (`data/scratch/uploads/`) with unique UUIDs. Once temporal inference is completed and translation results are returned, temporary video files are automatically unlinked and permanently deleted unless the user explicitly opts into authorized research feedback.
3. **No Credential Leakage:** No private API keys, cloud storage credentials, or private video URLs will ever be committed to version control. The repository provides a clean `.env.example` template.
