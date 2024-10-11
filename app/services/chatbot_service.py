from app.models.qa_model import QAModel
from app.utils.data_loader import load_and_split_data
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, data_path):
        texts = load_and_split_data(data_path)
        self.qa_model = QAModel(texts)
        self.chat_history = {}
        logger.info("ChatbotService inicializado")

    def process_message(self, from_phone, message):
        logger.info(f"Procesando mensaje de {from_phone}: {message}")
        if from_phone not in self.chat_history:
            self.chat_history[from_phone] = []
        
        try:
            answer = self.qa_model.get_answer(message)
            self.chat_history[from_phone].append((message, answer))
            logger.info(f"Respuesta generada para {from_phone}: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {str(e)}", exc_info=True)
            return "Lo siento, ocurrió un error al procesar tu mensaje. Por favor, intenta de nuevo."

