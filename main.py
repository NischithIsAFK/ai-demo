from langchain_community.llms import Bedrock
import boto3
import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
# Model ID
modelID = "anthropic.claude-v2:1"


# Function to interact with Bedrock
def my_chatbot(freeform_text):
    payload = {
        "modelId": modelID,
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps(
            {
                "prompt": f"\n\nHuman: {freeform_text}\n\nAssistant:",
                "max_tokens_to_sample": 300,
                "temperature": 0.5,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31",
            }
        ),
    }

    response = bedrock_client.invoke_model(
        modelId=payload["modelId"],
        contentType=payload["contentType"],
        accept=payload["accept"],
        body=payload["body"],
    )

    # Decode and return response
    response_body = json.loads(response["body"].read().decode("utf-8"))
    return response_body["completion"]


# Streamlit UI
st.title("Bedrock Chatbot")

freeform_text = st.sidebar.text_area(label="What is your question?", max_chars=100)

if freeform_text:
    response = my_chatbot(freeform_text)
    st.write(response)
