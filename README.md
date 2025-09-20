# 🚀 Contrastive Learning Project Plan

## 1. Objectives

* Implement and train a contrastive learning encoder (SimCLR baseline) on **ImageNet-100**.
* Extend to **mini-CLIP** (aligning image ↔ text embeddings, supporting zero-shot classification).
* Build a **retrieval system demo** (Faiss + Streamlit, supporting image-to-image and text-to-image search).
* Conduct **ablation studies & visualization** (augmentation strategies, temperature scaling, queue size).

---

## 2. Development Phases & Core Tasks

| Phase                                | Timeline (Est.) | Core Tasks                                                                                                                                                       | Models / Methods                                   |
| ------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Phase 1: Setup**                   | Week 1–2        | - Project initialization (code structure, configs)<br>- Data loading (ImageNet-100)<br>- Implement data augmentation pipeline (crop, color jitter, blur, flip)   | `torchvision.transforms`, DataLoader               |
| **Phase 2: Baseline**                | Week 3–4        | - Implement SimCLR model (ResNet-50 encoder + MLP head)<br>- Train & evaluate with linear probe                                                                  | ResNet-50, NT-Xent loss                            |
| **Phase 3: Multimodal Extension**    | Week 5–6        | - Add text encoder (MiniLM / DistilBERT)<br>- Train mini-CLIP (image embeddings ↔ class text embeddings)<br>- Test zero-shot classification                      | SimCLR encoder + MiniLM encoder + Contrastive loss |
| **Phase 4: Retrieval Application**   | Week 7          | - Build embedding index with Faiss<br>- Implement image-to-image and text-to-image retrieval<br>- Deploy Streamlit demo                                          | Faiss, Streamlit                                   |
| **Phase 5: Expansion & Analysis**    | Week 8–9        | - Ablation studies (augmentation combinations, temperature, queue size)<br>- Robustness evaluation (noise, blur)<br>- Visualization of embeddings (t-SNE / UMAP) | Ablation configs, sklearn.manifold (t-SNE), UMAP   |
| **Phase 6: Finalization & Delivery** | Week 10         | - Summarize results & visualizations<br>- Write technical report<br>- Extract résumé highlights<br>- Release codebase & demo                                     | Markdown report, GitHub repo                       |

---

## 3. Models & Methods Summary

* **Encoder**: ResNet-50 (`torchvision.models`)
* **Projection Head**: 2-layer MLP (hidden=2048, output=128)
* **Loss Functions**: NT-Xent (SimCLR), InfoNCE (for CLIP alignment)
* **Text Encoders**: MiniLM / DistilBERT (HuggingFace Transformers)
* **Retrieval Tools**: Faiss (IndexFlatIP / HNSW)
* **Visualization**: t-SNE, UMAP, matplotlib / seaborn
* **Demo Framework**: Streamlit WebApp

---

