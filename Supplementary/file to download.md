Here are the required datasets and checkpoints with **full visible links** formatted in a table:

### Required Artifacts (InfoSeek Pipeline)

| Component | Artifact Name | Size | Full Visible Link / Download Source | CLI Download Command |
| :--- | :--- | :--- | :--- | :--- |
| **VQA Queries** | InfoSeek `test.csv` | ~MB | `https://drive.google.com/file/d/1cSG_dVuao9lKZy8vaUDWEo7mIHowjUeE/view?usp=drive_link` | `gdown 1cSG_dVuao9lKZy8vaUDWEo7mIHowjUeE` |
| **Knowledge Base** | InfoSeek 100K KB (`infoseek_100k_wiki.zip`) | ~1–2 GB | `https://huggingface.co/datasets/BByrneLab/Infoseek-EchoSight-100K-KB/resolve/main/infoseek_100k_wiki.zip` | `wget https://huggingface.co/datasets/BByrneLab/Infoseek-EchoSight-100K-KB/resolve/main/infoseek_100k_wiki.zip` |
| **Step 1 Index** | InfoSeek Summary FAISS Index | 287 MB | `https://drive.google.com/file/d/1CE-SyPNKEx6fT09LXoMh4ckmh7jfzG_E/view?usp=drive_link` | `gdown 1CE-SyPNKEx6fT09LXoMh4ckmh7jfzG_E` |
| **Step 2b Model** | Multimodal Fusion Reranker Weights (`reranker_model.pth.zip`) | 2.3 GB | `https://drive.google.com/file/d/1K1Mxn_ePj1cCHdbgESK8Wz_Xrw1bhBEM/view?usp=drive_link` | `gdown 1K1Mxn_ePj1cCHdbgESK8Wz_Xrw1bhBEM` |
| **Wiki Image Meta** | Wiki Images CSVs (13 split CSVs) | 269 MB | `https://drive.google.com/file/d/1__laja2XMKA-J3oBT7EFdLxLCEGfpw4r/view?usp=drive_link` | `gdown 1__laja2XMKA-J3oBT7EFdLxLCEGfpw4r` |
| **Query Images** | Oven Image Downloader | Dynamic | `https://github.com/edchengg/oven_eval/tree/main/image_downloads` | *(Download selectively for chosen test IDs)* |
| **Step 3 Generator** | Pretrained LLM / MLLM (Hugging Face) | ~5–8 GB | `https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`<br>`https://huggingface.co/OpenGVLab/InternVL2_5-8B` | *(Load directly via `transformers`)* |

---

### Skipped / Not Needed Artifacts (For Reference)

| Component | Artifact Name | Size | Original Link | Reason Skipped |
| :--- | :--- | :--- | :--- | :--- |
| **VQA Train** | InfoSeek `train.csv` | ~MB | `https://drive.google.com/file/d/1cQiQmdFq8_8gsaZPsmzKzIjZhdcd_kxP/view?usp=drive_link` | No retraining needed |
| **E-VQA Queries** | E-VQA `train.csv` | ~MB | `https://drive.google.com/file/d/13BZZAserLlqKT_RHq4sX5NAIu7Ii77eE/view?usp=drive_link` | Skipping E-VQA route |
| **E-VQA Queries** | E-VQA `val.csv` | ~MB | `https://storage.googleapis.com/encyclopedic-vqa/val.csv` | Skipping E-VQA route |
| **E-VQA Queries** | E-VQA `test.csv` | ~MB | `https://storage.googleapis.com/encyclopedic-vqa/test.csv` | Skipping E-VQA route |
| **E-VQA KB** | E-VQA 2M KB (`encyclopedic_kb_wiki.zip`) | 4.9 GB | `https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip` | 50GB+ ecosystem; slow downloads |
| **E-VQA Index** | E-VQA Summary FAISS Index | 5.6 GB | `https://drive.google.com/file/d/1FPx1EOTjcx8zXPoPD9fBQ6cbucF0fL_D/view?usp=drive_link` | Using InfoSeek 287 MB FAISS instead |
| **Step 2a Data** | Hard Negative Training Samples | 159 MB | `https://drive.google.com/file/d/1X9sOZV5jSielgszfIcncvcoCtDumHkBD/view?usp=drive_link` | Reranker is already pretrained |
| **Step 3 Finetuned** | Finetuned E-VQA LLaVA-1.5-7B | 12 GB | `https://drive.google.com/file/d/17gaVltG2acdWKyG8oBKsrvUURNnqVZGD/view?usp=drive_link` | Fine-tuned for E-VQA, not InfoSeek |
| **Dead KB Link** | OMGM GDrive InfoSeek 100K KB | N/A | `https://drive.google.com/file/d/1cIbKtYryD7XBAw0tjrrCvMCJC2rIzLM5/view?usp=drive_link` | Dead Google Drive link (mirrored on HF) |