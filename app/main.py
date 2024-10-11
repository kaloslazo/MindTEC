import logging
from fastapi import FastAPI, Request
from app.services.twilio_service import sendMessage as send_whatsapp_message
from app.services.chatbot_service import ChatbotService
from openai import OpenAIError  

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

chatbot_service = ChatbotService("app/data/syllabus_data.csv")

@app.post("/hook")
async def chat(request: Request):
    form_data = await request.form()
    body_data = form_data.get("Body", "")
    from_phone = form_data.get("From", "")
    
    logger.info(f"Mensaje recibido de {from_phone}: {body_data}")
    
    try:
        response = chatbot_service.process_message(from_phone, body_data)
        logger.debug(f"Respuesta generada: {response}")
        send_result = send_whatsapp_message(from_phone, response)
        return {"status": "success", "message": send_result}
    
    except OpenAIError as e:
        logger.error(f"Error de OpenAI: {str(e)}")
        error_message = "Lo siento, estamos experimentando problemas técnicos. Por favor, intenta de nuevo más tarde."
        send_whatsapp_message(from_phone, error_message)
        return {"status": "error", "message": "Error de OpenAI", "details": str(e)}
    
    except Exception as e:
        logger.error(f"Error inesperado al procesar el mensaje: {str(e)}", exc_info=True)
        error_message = "Lo siento, ocurrió un error inesperado. Por favor, intenta de nuevo más tarde."
        send_whatsapp_message(from_phone, error_message)
        return {"status": "error", "message": "Error interno", "details": str(e)}
