import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    TWILIO_SMS_PHONE_NUMBER = os.getenv('TWILIO_SMS_PHONE_NUMBER')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Configuraciones actualizadas para Qdrant
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')

config = Config()
