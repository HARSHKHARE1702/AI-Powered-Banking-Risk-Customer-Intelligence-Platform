import os
from dotenv import load_dotenv
from openai import OPENAI
load_dotenv()
client = OPENAI(
  api_key = os.getenv("OPENAI API KEY")
)
MODEL = os.getenv("OPENAI MODEL","gpt-6-astra")
def generate_response (prompt:str) -> str :
  response = client.responses.create(
    model = MODEL,
    instructions = (
      "You are a banking AI assistant. "
            "Provide accurate, clear and professional answers. "
            "Do not invent banking policies or financial facts."
    ),
    return response.output_text
