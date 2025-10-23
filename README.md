This project performs sentiment analysis on customer reviews from the Telco Customer Churn Dataset and automatically generates personalized replies to each review using advanced NLP models.

The system is designed for telecom customer feedback management — helping companies understand sentiment trends and respond intelligently to customers in real time.

Project Overview :

Sentiment Analysis on customer reviews using a fine-tuned transformer model.

Automated Customer Replies generated with an LLM for professional, context-aware responses.

FastAPI-based API for real-time interaction and easy integration with dashboards or customer support systems.

MLflow + DagsHub Integration for experiment tracking.

Models Used :

Sentiment Classification	tabularisai/robust-sentiment-analysis	Hugging Face
Auto Reply Generation	meta-llama/Llama-3.2-1B-Instruct	Meta LLaMA

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
