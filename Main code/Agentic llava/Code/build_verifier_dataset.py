"""
Build Pairwise Visual Verifier Fine-Tuning Datasets (Symmetric Pair Expansion)

Paper: Agentic KB-VQA with Adaptive Visual Verification
Method: For each sampled query, generate two symmetric instruction pairs:
  - Instance A: Option (A) is Ground Truth, Option (B) is Hard Negative -> Target is (A)
  - Instance B: Option (A) is Hard Negative, Option (B) is Ground Truth -> Target is (B)
This guarantees exactly 50% target (A) and 50% target (B), completely neutralizing
the severe Option-(A) primacy bias observed in zero-shot VLMs.
"""

import os
import json
import urllib.parse
import pandas as pd
import numpy as np

def url_to_title(url: str) -> str:
    """Extract clean Wikipedia title from Wikipedia URL."""
    slug = str(url).rstrip('/').split('/')[-1]
    return urllib.parse.unquote(slug).replace('_', ' ').strip()

def build_llava_prompt(option_a: str, option_b: str) -> str:
    """Generate exact prompt used in agentic controller."""
    return (
        "USER: <image>\n"
        "Look at the image carefully.\n"
        "Which entity does this image depict?\n"
        f"Option (A): {option_a}\n"
        f"Option (B): {option_b}\n"
        "Answer strictly with (A) or (B):\n"
        "ASSISTANT:"
    )

def main():
    np.random.seed(42)
    output_dir = "others/files/fine_tuning_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading official InfoSeek train parquet from Hugging Face mirror...")
    url = "https://huggingface.co/datasets/BByrneLab/Infoseek-train-with-EvaClip-retrieval-top5/resolve/main/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(url)
    print(f"   Loaded {len(df)} total training rows.")

    # Exclude any 1000 eval queries if present (just to guarantee 100% disjointness)
    eval_csv_path = "pre helper/files/infoseek_test_sample_1000.csv"
    if os.path.exists(eval_csv_path):
        eval_df = pd.read_csv(eval_csv_path)
        eval_qids = set(eval_df['data_id'])
        eval_imgs = set(eval_df['dataset_image_ids'])
        df = df[~df['question_id'].isin(eval_qids)]
        print(f"   Filtered out eval queries. Remaining pool: {len(df)}")

    # Shuffle deterministically
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_train_queries = 3000
    n_val_queries = 300
    total_needed = n_train_queries + n_val_queries

    print(f"2. Extracting {n_train_queries} train queries and {n_val_queries} validation queries with hard negatives...")

    records = []
    for _, row in df_shuffled.iterrows():
        gt_title = row['wiki_title']
        if not gt_title or pd.isna(gt_title):
            continue
        gt_title = str(gt_title).strip()
        gt_url = str(row['wikipedia_url'])
        pos_items = set(row['pos_item_ids']) if row['pos_item_ids'] is not None else {gt_url}

        # Find top hard negative from EvaClip retrieval candidates
        retrieved = row['retrieved_passage']
        neg_title = None
        neg_score = None
        neg_url = None

        for p in retrieved:
            pid = p['passage_id']
            title = url_to_title(pid)
            if pid not in pos_items and title.lower() != gt_title.lower() and len(title) > 0:
                neg_title = title
                neg_score = float(p.get('score', 0.0))
                neg_url = pid
                break

        if neg_title is None:
            continue

        # Clean image id and filename
        raw_img = str(row['img_path'])
        clean_img_id = raw_img.replace('.JPEG', '').replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
        img_filename = f"{clean_img_id}.jpg"

        records.append({
            'data_id': row['question_id'],
            'image_id': clean_img_id,
            'image_filename': img_filename,
            'question': str(row['question']).strip(),
            'gt_title': gt_title,
            'gt_url': gt_url,
            'neg_title': neg_title,
            'neg_url': neg_url,
            'neg_retrieval_score': neg_score,
            'gold_answer': str(row.get('gold_answer', ''))
        })

        if len(records) >= total_needed:
            break

    print(f"   Collected {len(records)} high-quality queries with verified hard negatives.")
    train_queries = records[:n_train_queries]
    val_queries = records[n_train_queries:n_train_queries + n_val_queries]

    # Save base query CSVs
    pd.DataFrame(train_queries).to_csv(os.path.join(output_dir, "sampled_train_queries_3000.csv"), index=False)
    pd.DataFrame(val_queries).to_csv(os.path.join(output_dir, "sampled_val_queries_300.csv"), index=False)
    print("   Saved sampled_train_queries_3000.csv and sampled_val_queries_300.csv")

    # 3. Symmetric Pair Expansion (Double size, 50% target A, 50% target B)
    def expand_symmetric_pairs(query_list, split_name: str):
        csv_rows = []
        json_conversations = []

        for q in query_list:
            qid = q['data_id']
            img = q['image_filename']
            gt = q['gt_title']
            neg = q['neg_title']
            question = q['question']

            # --- Pair 1: Option A is Ground Truth, Option B is Negative (Target: A) ---
            prompt_a = build_llava_prompt(option_a=gt, option_b=neg)
            target_letter_a = "(A)"
            target_text_a = f"(A) {gt}"

            csv_rows.append({
                'pair_id': f"{qid}_tgtA",
                'base_data_id': qid,
                'image_filename': img,
                'image_id': q['image_id'],
                'question': question,
                'option_a': gt,
                'option_b': neg,
                'ground_truth_entity': gt,
                'target_choice': target_letter_a,
                'target_text': target_text_a,
                'prompt': prompt_a,
                'neg_score': q['neg_retrieval_score']
            })

            json_conversations.append({
                "id": f"{qid}_tgtA",
                "image": img,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nLook at the image carefully.\nWhich entity does this image depict?\n"
                                 f"Option (A): {gt}\nOption (B): {neg}\nAnswer strictly with (A) or (B):"
                    },
                    {
                        "from": "gpt",
                        "value": target_text_a
                    }
                ]
            })

            # --- Pair 2: Option A is Negative, Option B is Ground Truth (Target: B) ---
            prompt_b = build_llava_prompt(option_a=neg, option_b=gt)
            target_letter_b = "(B)"
            target_text_b = f"(B) {gt}"

            csv_rows.append({
                'pair_id': f"{qid}_tgtB",
                'base_data_id': qid,
                'image_filename': img,
                'image_id': q['image_id'],
                'question': question,
                'option_a': neg,
                'option_b': gt,
                'ground_truth_entity': gt,
                'target_choice': target_letter_b,
                'target_text': target_text_b,
                'prompt': prompt_b,
                'neg_score': q['neg_retrieval_score']
            })

            json_conversations.append({
                "id": f"{qid}_tgtB",
                "image": img,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nLook at the image carefully.\nWhich entity does this image depict?\n"
                                 f"Option (A): {neg}\nOption (B): {gt}\nAnswer strictly with (A) or (B):"
                    },
                    {
                        "from": "gpt",
                        "value": target_text_b
                    }
                ]
            })

        # Save CSV and JSON
        df_out = pd.DataFrame(csv_rows)
        # Interleave or shuffle so A and B alternate evenly
        df_out = df_out.sample(frac=1.0, random_state=42).reset_index(drop=True)
        csv_file = os.path.join(output_dir, f"verifier_{split_name}_symmetric_{len(df_out)}.csv")
        df_out.to_csv(csv_file, index=False)

        json_file = os.path.join(output_dir, f"verifier_{split_name}_symmetric_{len(json_conversations)}.json")
        with open(json_file, 'w') as f:
            json.dump(json_conversations, f, indent=2)

        # Check balance
        counts = df_out['target_choice'].value_counts().to_dict()
        print(f"   [{split_name.upper()}] Generated {len(df_out)} symmetric pairs.")
        print(f"     Target distribution: {counts} (Balance: {counts.get('(A)', 0)} vs {counts.get('(B)', 0)})")
        print(f"     Saved: {csv_file}")
        print(f"     Saved: {json_file}")

        return df_out

    print("\n3. Generating Symmetric Training Dataset (6,000 instances)...")
    expand_symmetric_pairs(train_queries, split_name="train")

    print("\n4. Generating Symmetric Validation Dataset (600 instances)...")
    expand_symmetric_pairs(val_queries, split_name="val")

    # 5. Extract unique image IDs for downloading/extracting
    all_queries = train_queries + val_queries
    unique_image_ids = sorted(list(set(q['image_id'] for q in all_queries)))
    image_list_path = os.path.join(output_dir, "required_oven_images_3300.txt")
    with open(image_list_path, 'w') as f:
        for img_id in unique_image_ids:
            f.write(f"{img_id}.jpg\n")
    print(f"\n5. Unique image IDs needed: {len(unique_image_ids)}")
    print(f"   Saved image extraction list to: {image_list_path}")
    print("\nAll fine-tuning datasets successfully constructed!")

if __name__ == "__main__":
    main()
