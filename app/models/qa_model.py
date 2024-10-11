import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from app.config import config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Eres un asistente virtual para estudiantes de la Universidad de Ingeniería y Tecnología (UTEC). Tu tarea es proporcionar información precisa basada en el contenido de los sílabos de los cursos.

Contexto del sílabo:
{context}

Pregunta del estudiante: {question}

Instrucciones:
1. Usa la información proporcionada en el contexto anterior para responder.
2. Si se pregunta por referencias bibliográficas, busca específicamente una sección llamada "BIBLIOGRÁFICAS" o similar en el contexto.
3. Si encuentras referencias bibliográficas relevantes, menciónalas directamente.
4. Si la información exacta no está en el contexto, pero hay información parcial o relacionada, proporciona esa información y menciona que es parcial.
5. Si no hay absolutamente ninguna información relevante, di "Lo siento, no tengo información específica sobre eso en el sílabo."
6. No inventes ni inferas información que no esté explícitamente en el contexto.

Respuesta basada en la información del sílabo:
"""

class QAModel:
    def __init__(self, texts):
        logger.info(f"QAModel inicializado con {len(texts)} documentos")
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=[
            "context", "question"])

        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2")
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        self.qdrant = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

        self.llm = OpenAI(temperature=0.3, max_tokens=300)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.qdrant.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def create_collection_if_not_exists(self):
        collections = self.qdrant_client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            logger.info(f"Creando nueva colección: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE),
            )
        else:
            logger.info(f"La colección {self.collection_name} ya existe")

    def split_content(self, content, max_length=500):
        sections = []
        current_section = ""
        for line in content.split('\n'):
            if len(current_section) + len(line) > max_length:
                sections.append(current_section.strip())
                current_section = line
            else:
                current_section += "\n" + line
        if current_section:
            sections.append(current_section.strip())
        return sections

    def load_documents(self, texts):
        logger.info(f"Cargando {len(texts)} documentos en Qdrant")
        points = []
        for i, text in enumerate(texts):
            content = text.page_content
            vector = self.embeddings.embed_query(content)
            point = PointStruct(
                id=str(i),
                payload={'text': content, 'metadata': text.metadata},
                vector=vector
            )
            points.append(point)
        
        try:
            operation_info = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Operación de carga completada. Info: {operation_info}")
        except Exception as e:
            logger.error(f"Error al cargar documentos en Qdrant: {e}")
            raise
        
        logger.info("Documentos cargados exitosamente en Qdrant")

    def getAnswer(self, question):
        logger.info(f"Procesando pregunta: {question}")
        if len(question.split()) < 3:
            return "🤔 Por favor, proporciona más detalles para poder ayudarte mejor."
        try:
            logger.debug("Iniciando búsqueda en Qdrant")
            
            # Realizar la búsqueda directamente en Qdrant
            query_vector = self.embeddings.embed_query(question)
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            full_context = []
            for i, result in enumerate(search_results):
                logger.debug(f"Documento {i+1}:")
                logger.debug(f"ID: {result.id}, Score: {result.score}")
                logger.debug(f"Contenido: {result.payload['text'][:200]}...")
                full_context.append(f"Documento {i+1}:\n{result.payload['text']}")
            
            context = "\n\n".join(full_context)
            logger.debug(f"Contexto completo pasado al modelo:\n{context}")
            
            # Usar el contexto completo para generar la respuesta
            prompt = self.qa_chain.combine_docs_chain.llm_chain.prompt.format(
                context=context,
                question=question
            )
            response = self.qa_chain.combine_docs_chain.llm_chain.llm.predict(prompt)
            
            logger.debug(f"Respuesta generada: {response}")
            return response
        except Exception as e:
            logger.error(f"Error al procesar la pregunta: {str(e)}", exc_info=True)
            return "🙁 Lo siento, tuve un pequeño problema al procesar tu pregunta. ¿Podrías intentar reformularla?"

    def test_retrieval(self, query):
        logger.info(f"Probando recuperación para la consulta: {query}")
        query_vector = self.embeddings.embed_query(query)
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=3
        )
        logger.info("Resultados de búsqueda directa en Qdrant:")
        for result in search_result:
            logger.info(f"ID: {result.id}, Score: {result.score}")
            logger.info(f"Contenido: {result.payload['text'][:200]}...")

        # Probar la cadena completa
        qa_result = self.getAnswer(query)
        logger.info(f"Respuesta del modelo: {qa_result}")
