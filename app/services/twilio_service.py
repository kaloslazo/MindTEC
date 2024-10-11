from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from app.config import config
import logging

logger = logging.getLogger(__name__)

client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

def sendWhatsappMessage(to_phone, body_data):
    if to_phone.startswith('whatsapp:'): to_phone = to_phone[9:].strip()
    if not to_phone.startswith('+'): to_phone = '+' + to_phone
    
    try:
        message = client.messages.create(
            from_=f'whatsapp:{config.TWILIO_PHONE_NUMBER}',
            body=body_data,
            to=f'whatsapp:{to_phone}'
        )
        
        logger.info(f"Mensaje enviado. SID: {message.sid}")
        return "Mensaje enviado correctamente"
    
    except TwilioRestException as e:
        logger.error(f"Error al enviar mensaje: {str(e)}")
        raise
