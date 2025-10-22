from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch, re, time, os, string
import mlflow, dagshub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from app.config import MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dagshub.init(repo_owner="ChaitanyaC20", repo_name="Sentiment_Analysis", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
sentiment_model.eval()

llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_tokenizer.padding_side = "left"
llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
).eval()

class ReviewRequest(BaseModel):
    review: str

def is_meaningful_text(text: str) -> bool:
    cleaned = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text).strip()
    return bool(re.search(r"[A-Za-z]{2,}", cleaned))

def predict_sentiment(review: str) -> tuple[int, float]:
    inputs = sentiment_tokenizer(review, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax()) + 1
    confidence = float(probs.max())
    return pred, confidence

@torch.inference_mode()
def generate_reply(review: str, score: int) -> str:
    tone = "neutral and professional"
    if score <= 2:
        tone = "apologetic and helpful"
    elif score == 3:
        tone = "neutral and informative"
    elif score >= 4:
        tone = "grateful and positive"
    prompt = (
        f"Dear Customer, as telecom support, write a short, {tone} reply (under 80 words). "
        f"Use 'We' not 'I'. Customer review: {review}\nReply:"
    )

    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llama_model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llama_tokenizer.eos_token_id,
        eos_token_id=llama_tokenizer.eos_token_id
    )

    reply = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.split("Reply:")[-1].strip()
    reply = re.sub(r"\s+", " ", reply)
    reply = re.sub(r'(\.|\!|\?)\s*[^\.!\?]*$', r'\1', reply)
    if not reply.endswith((".", "!", "?")):
        reply += "."
    return reply

@router.post("/analyze_and_reply")
def analyze_and_reply(request: ReviewRequest):
    review = request.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Empty review text")
    if not is_meaningful_text(review):
        return {
            "predicted_sentiment": None,
            "auto_reply": "We couldn't understand your feedback. Could you please rephrase or provide more details?",
            "latency_seconds": 0
        }
    try:
        start_time = time.time()
        score, confidence = predict_sentiment(review)
        
        reply = generate_reply(review, score)
        latency = round(time.time() - start_time, 2)

        with mlflow.start_run(run_name="analyze_and_reply_api", nested=True):
            mlflow.log_param("review_text", review[:100])
            mlflow.log_param("predicted_score", score)
            mlflow.log_param("confidence", confidence)
            mlflow.log_param("reply_text", reply)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_param("sentiment_model", MODEL_DIR)
            mlflow.log_param("reply_model", llama_model_name)

        return {
            "predicted_sentiment": score,
            "confidence": round(confidence, 3),
            "auto_reply": reply,
            "latency_seconds" : latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))