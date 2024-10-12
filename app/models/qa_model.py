import logging
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from app.config import config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Eres un asistente virtual para estudiantes de la Universidad de Ingeniería y Tecnología (UTEC). Tu tarea es proporcionar información precisa y relevante basada en el contenido de los sílabos de los cursos, promociones disponibles y actividades deportivas. Para mejorar la claridad y efectividad de las respuestas, sigue estas directrices estrictamente:

- No uses triple asterisco (*) en ningún caso.
- Cuando quieras colocar doble asterisco (*) pon únicamente uno ().
- Usa un solo asterisco () para resaltar información importante, ejemplo: *nota importante.
- Usa guion bajo () para términos técnicos o conceptos clave, ejemplo: _algoritmo.
- Usa "comillas" para citas textuales, ejemplo: "texto citado".
- Utiliza un solo emoji al final de la respuesta para hacerla más amigable, cuando sea apropiado.
- Usa listas con guiones (-) para enumerar elementos, prohibido usar ** o * en guiones.
- Para títulos o subtítulos, usa un solo asterisco (*) al inicio de la línea, sin asterisco al final.
- Para subtítulos dentro de listas, no uses formato especial, solo el guion (-) al inicio.

Contexto:
{context}

Pregunta del estudiante: {question}

Instrucciones:
1. Identifica el tipo de información solicitada (sílabo, promoción o actividad deportiva).
2. Extrae y usa información directamente del contexto proporcionado para formular la respuesta.
3. Para sílabos:
   - Si se pregunta por referencias bibliográficas, menciónalas directamente.
   - Incluye detalles relevantes como créditos, modalidad, y otros datos importantes del curso si la pregunta los relaciona.
4. Para promociones:
   - Si la pregunta es general, proporciona una lista concisa de categorías y nombres de establecimientos, sin descripciones ni detalles adicionales.
   - Usa el formato: "Categoría: Nombre1, Nombre2".
   - Limita la respuesta a un máximo de 5 categorías y 3 nombres por categoría.
   - Al final, pregunta si el usuario desea más información sobre alguna promoción específica.
5. Para actividades deportivas: [Se mantiene igual]
6. Si la información solicitada no está disponible, responde con: "Lo siento, no tengo información específica sobre eso."
7. Evita inferir información que no esté explícitamente presente en el contexto proporcionado.
8. Asegúrate de que la respuesta sea clara, bien organizada y útil para el estudiante.
9. No usar numeración con #

Respuesta basada en la información proporcionada:
"""

class QAModel:
    def __init__(self, texts):
        logger.info(f"QAModel inicializado con {len(texts)} documentos")
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        self.qdrant = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=300)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.qdrant.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        self.clear_collection()
        self.load_documents(texts)

    def clear_collection(self):
        logger.info(f"Limpiando la colección {self.collection_name}")
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Colección {self.collection_name} eliminada")
        except Exception as e:
            logger.warning(f"Error al eliminar la colección: {e}")
        
        self.create_collection_if_not_exists()

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
        for text in texts:
            content = text.page_content
            vector = self.embeddings.embed_query(content)
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generamos un UUID único para cada punto
                payload={
                    'text': content, 
                    'metadata': text.metadata,
                    'doc_type': text.metadata.get('type', 'unknown')
                },
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
        
            if "promociones" in question.lower() and "universidad" in question.lower():
                query_filter = {"must": [{"key": "doc_type", "match": {"value": "promo"}}]}
            elif any(word in question.lower() for word in ["deporte", "cancha", "reserva"]):
                query_filter = {"must": [{"key": "doc_type", "match": {"value": "deporte"}}]}
            else:
                query_filter = None

            query_vector = self.embeddings.embed_query(question)
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=5
            )
        
            full_context = []
            for i, result in enumerate(search_results):
                logger.debug(f"Documento {i+1}:")
                logger.debug(f"ID: {result.id}, Score: {result.score}")
                logger.debug(f"Tipo: {result.payload.get('doc_type', 'unknown')}")
                logger.debug(f"Contenido: {result.payload['text'][:200]}...")
                full_context.append(f"Documento {i+1} ({result.payload.get('doc_type', 'unknown')}):\n{result.payload['text']}")
        
            context = "\n\n".join(full_context)
            logger.debug(f"Contexto completo pasado al modelo:\n{context}")
        
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
