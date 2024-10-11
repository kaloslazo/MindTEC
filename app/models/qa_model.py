from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from app.config import config
from qdrant_client import QdrantClient

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Responde brevemente basándote en esta información del sílabo del curso preguntado por el estudiante:
{context}

Pregunta: {question}

Respuesta muy concisa:
"""

class QAModel:
    def __init__(self, texts):
        logger.info(f"QAModel inicializado con {len(texts)} documentos")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        self.qdrant = Qdrant(
            client = self.qdrant_client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embeddings=self.embeddings,
        )
        self.llm = OpenAI(temperature=0, max_tokens=100)
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.qdrant.as_retriever(search_kwargs={"k": 1}),
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    def get_answer(self, question):
        logger.info(f"Procesando pregunta: {question}")
        if len(question.split()) < 3:
            return "Por favor, haz una pregunta más específica sobre cualquier curso."
        try:
            logger.debug("Iniciando búsqueda en Qdrant")
            result = self.qa_chain({"question": question, "chat_history": []})
            logger.debug(f"Respuesta generada: {result['answer']}")
            return result['answer']
        except Exception as e:
            logger.error(f"Error al procesar la pregunta: {str(e)}", exc_info=True)
            return "Lo siento, no pude procesar tu pregunta. Por favor, intenta reformularla."
