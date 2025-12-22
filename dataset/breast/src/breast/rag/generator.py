from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
        model_name='openai/gpt-oss-20b',
        temperature=0.6,
        api_key= groq_api_key
    )

def generate_medical_guidance(probability, context):
    prompt = f"""
    You are an AI assistant providing EDUCATIONAL MEDICAL INFORMATION.
    You are NOT diagnosing or prescribing.

    Tumor probability: {probability*100:.2f}%

    Using the medical context below, provide:
    - General educational explanation
    - Possible next steps for learning
    - Clear medical disclaimer

    Medical context:
    {context}
    """

    response = model.invoke(prompt)
    return response.content