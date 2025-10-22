import os
import re
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
import mlflow
import dagshub
from app.config import (DATA_PATH, RESULTS_PATH, MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, ARTIFACT_DIR)
from app.data import load_sentiment_data
from app.preprocess_sentiment import clean_review

dagshub.init(repo_owner="ChaitanyaC20", repo_name="Sentiment_Analysis", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    try:
        login(HF_TOKEN)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Hugging Face login skipped: {e}")

df = load_sentiment_data()
df["Cleaned_Reviews"] = df["Generated_Reviews"].fillna("").apply(clean_review)
df = df[df["Cleaned_Reviews"].str.strip() != ""]
print(f"{len(df)} reviews ready for reply generation.")

model_name = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading reply model '{model_name}' on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

torch.backends.cudnn.benchmark = True
print("Model loaded successfully on", "GPU" if torch.cuda.is_available() else "CPU")

@torch.inference_mode()
def generate_replies(reviews, scores, max_new_tokens=80):
    prompts, tones = [], []
    for review, score in zip(reviews, scores):
        tone = "neutral and professional"
        if isinstance(score, (int, float)):
            if score <= 2:
                tone = "apologetic and helpful"
            elif score == 3:
                tone = "neutral and informative"
            elif score >= 4:
                tone = "grateful and positive"
        tones.append(tone)
        prompts.append(
            f"Dear Customer, as telecom support, write a short, {tone} reply (under 80 words). "
            f"Use 'We' not 'I'.  Customer review: {review}\nReply:"
        )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    replies = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        reply = text.split("Reply:")[-1].strip()
        reply = re.sub(r'\s+', ' ', reply)
        reply = re.sub(r'(\.|\!|\?)\s*[^\.!\?]*$', r'\1', reply)
        if not reply.endswith(('.', '!', '?')):
            reply += '.'
        replies.append(reply)

    return replies, tones

with mlflow.start_run(run_name="customer_reply_generation"):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("device", device.type)
    mlflow.log_param("num_reviews", len(df))
    mlflow.log_param("results_file", RESULTS_PATH)

    batch_size = 4
    save_every = 200
    output_path = RESULTS_PATH.replace(".xlsx", "_Auto_Replies.xlsx")

    start_time = time.time()
    all_replies, all_tones = [], []

    for i in tqdm(range(0, len(df), batch_size), ncols=100, desc="Reply Generation"):
        batch = df.iloc[i:i + batch_size]
        replies, tones = generate_replies(batch["Cleaned_Reviews"].tolist(), batch["Satisfaction_Score"].tolist())
        df.loc[batch.index, "Auto_Reply"] = replies
        df.loc[batch.index, "Reply_Tone"] = tones

        all_replies.extend(replies)
        all_tones.extend(tones)

        if i % save_every == 0 and i > 0:
            partial_path = output_path.replace(".xlsx", f"_partial_{i}.xlsx")
            df.to_excel(partial_path, index=False)
            mlflow.log_artifact(partial_path)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate / 60
            print(f"Autosaved up to {i} rows → ETA: {eta:.1f} min")

    df.to_excel(output_path, index=False)
    csv_path = output_path.replace(".xlsx", ".csv")
    df.to_csv(csv_path, index=False)

    mlflow.log_artifact(output_path)
    mlflow.log_artifact(csv_path)
    mlflow.log_param("final_output", output_path)

    avg_len = sum(len(r.split()) for r in all_replies) / len(all_replies)
    mlflow.log_metric("avg_reply_length", avg_len)
    mlflow.log_metric("total_replies", len(all_replies))

    print(f"\nReplies saved successfully → {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Average reply length: {avg_len:.1f} words")

    mlflow.end_run()