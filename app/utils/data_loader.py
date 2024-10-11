import csv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def truncateText(text, max_length=1000):
    return text[:max_length] if text else ""

def loadAndSplitData(file_path):
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Archivo: {row.get('Archivo', 'No especificado')}\n"
            content += f"Carrera: {row.get('Carrera', 'No especificada')}\n"
            content += f"Curso: {row.get('Curso', 'No especificado')}\n"
            content += f"Malla: {row.get('Malla', 'No especificada')}\n"
            content += f"Modalidad: {row.get('Modalidad', 'No especificada')}\n"
            content += f"Créditos: {row.get('Creditos', 'No especificados')}\n"
            content += f"Objetivos: {truncateText(row.get('Objetivos', ''))}\n"
            content += f"Competencias: {truncateText(row.get('Competencias', ''))}\n"
            content += f"Resultados de Aprendizaje: {truncateText(row.get('Resultados de Aprendizaje', ''))}\n"
            content += f"Temas: {truncateText(row.get('Temas', ''))}\n"
            content += f"Sistema de Evaluación: {truncateText(row.get('Sistema de Evaluación', ''))}\n"
            content += f"Referencias Bibliográficas: {truncateText(row.get('Referencias Bibliográficas', ''))}"

            documents.append(Document(page_content=content, metadata=row))

    logger.info(f"Número de documentos cargados: {len(documents)}")
    if documents:
        logger.info(f"Muestra del primer documento:\n{documents[0].page_content[:500]}...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    logger.info(f"Número de fragmentos de texto generados: {len(texts)}")
    if texts:
        logger.info(f"Muestra del primer fragmento:\n{texts[0].page_content[:500]}...")

    return texts
