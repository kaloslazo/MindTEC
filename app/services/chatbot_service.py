from app.models.qa_model import QAModel
from app.utils.data_loader import loadAndSplitData
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, data_paths):
        logger.info("ChatbotService inicializado")
        texts = loadAndSplitData(data_paths)
        self.qa_model = QAModel(texts)
        self.qa_model.test_retrieval("¿Me recomiendas alguna referencia bibliografica del curso de tendencias de mercado?")
        self.qa_model.test_retrieval("¿Qué beneficios hay en la categoría de restaurantes?")
        self.qa_model.test_retrieval("¿Cómo puedo reservar una cancha de fútbol?")
        self.chat_history = {}

    def processMessage(self, from_phone, message):
        logger.info(f"Procesando mensaje de {from_phone}: {message}")
        if from_phone not in self.chat_history:
            self.chat_history[from_phone] = []

        try:
            answer = self.qa_model.getAnswer(message)
            self.chat_history[from_phone].append((message, answer))
            logger.info(f"Respuesta generada para {from_phone}: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {str(e)}", exc_info=True)
            return "Lo siento, ocurrió un error al procesar tu mensaje. Por favor, intenta de nuevo."
