from fastapi import FastAPI
from app.reply_api import router as reply_router
import uvicorn

app = FastAPI(
    title="Sentiment Analysis & Auto Reply API",
    description="API that predicts sentiment and generates automatic replies using LLaMA.",
    version="1.0.0"
)

app.include_router(reply_router, prefix="/api", tags=["Sentiment + Reply"])

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)