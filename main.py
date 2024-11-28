from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import aiofiles
from model import BreastCancerModel
import requests
import os
from g4f.client import Client

app = FastAPI()
model = BreastCancerModel()

class PredictionResponse(BaseModel):
    prediction: str
    detailed_analysis: str

@app.post("/upload/", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...)):
    file_location = f"temp_images/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    async with aiofiles.open(file_location, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Predict using the local model
    prediction = model.predict(file_location)

    # Send the prediction to the LLM for detailed analysis
    detailed_analysis = await get_llm_analysis(prediction)

    return PredictionResponse(prediction=prediction, detailed_analysis=detailed_analysis)

# async def get_llm_analysis(prediction: str) -> str:
#     url = "https://api.openai.com/v1/completions"  # Replace with your LLM API endpoint
#     headers = {
#         "Authorization": f"Bearer sk-Aihp7xyS4qqfDJNfusHCkQN7tcnS0NAEvKcwydXk6hT3BlbkFJXmcpEOL57Kl5teP-kv6cPl-wT0jTC40MR_PQPQ3xcA",  # Replace with your OpenAI API key
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "text-davinci-003",  # Replace with your LLM model
#         "prompt": f"Provide a detailed analysis for a breast cancer diagnosis prediction: {prediction}",
#         "max_tokens": 150
#     }

#     response = requests.post(url, headers=headers, json=data)
#     response_data = response.json()
#     print("API Response:", response_data)
#     return response_data.get("choices", [{}])[0].get("text", "No detailed analysis available.")


async def get_llm_analysis(prediction: str) -> str:
    client = Client()
    # Define the prompt for the LLM
    prompt = f"Provide a detailed analysis for a breast cancer diagnosis prediction: {prediction}"

    # Send the request to g4f's API (note: g4f.client may be synchronous, so no async call)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the detailed analysis from the response
    detailed_analysis = response.choices[0].message.content if response else "No detailed analysis available."

    return detailed_analysis


# import openai
# async def get_llm_analysis(prediction: str) -> str:
#     # Set your public hosted OpenAI API key and base URL
#     openai.api_key = 'anything'  # Placeholder API key since it's not used with public hosted APIs
#     openai.api_base = "http://localhost:3040/v1/"  # Replace with your actual hosted API base URL

#     try:
#         # Make the chat completion request
#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"Provide a detailed analysis for a breast cancer diagnosis prediction: {prediction}"}
#             ],
#             max_tokens=150
#         )

#         # Extract the response content
#         return completion.choices[0].message['content']

#     except Exception as e:
#         print(f"Error during API call: {e}")
#         return "No detailed analysis available."

