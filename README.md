This project performs sentiment analysis on customer reviews from the Telco Customer Churn Dataset and automatically generates personalized replies to each review using advanced NLP model.

The system is designed for telecom customer feedback management — helping companies understand sentiment trends and respond intelligently to customers in real time.

Project Overview :

Sentiment Analysis on customer reviews using a fine-tuned transformer model.

Automated Customer Replies generated with an LLM for professional, context-aware responses.

FastAPI-based API for real-time interaction and easy integration.

MLflow + DagsHub Integration for experiment tracking.

Models Used :

Sentiment Classification	tabularisai/robust-sentiment-analysis	Hugging Face
Auto Reply Generation	meta-llama/Llama-3.2-1B-Instruct	Meta LLaMA

Data Processing :

The dataset is loaded and cleaned using various NLP libraries to remove redundant characters, punctuation, and stopwords during preprocessing. After cleaning, the sentiment analysis model predicts sentiment  for each review, and the LLM generates a personalized, context-aware reply.

Installation : 

1) Clone the Repository : 
git clone https://github.com/ChaitanyaC20/NLP-on-Customer-churn.git
cd NLP-on-Customer-churn

2) Create and Activate Environment :
python -m venv py310env
py310env\Scripts\activate

3) Install Requirements :
pip install -r requirements.txt

4) Training the Sentiment Model : 
python -m app.train_sentiment

5) Generating Customer Replies
python -m app.reply

6) Running the API : 
uvicorn main:app --reload

Results :

<img width="1370" height="792" alt="image" src="https://github.com/user-attachments/assets/a1f4ff38-a1c1-4c90-a723-7286a75e4c18" />

<img width="1352" height="763" alt="image" src="https://github.com/user-attachments/assets/e89fd971-d12c-477a-97d9-32ecd14c2a93" />

<img width="1378" height="722" alt="image" src="https://github.com/user-attachments/assets/a0c22bfb-25dd-421c-9562-7026697e1a97" />



