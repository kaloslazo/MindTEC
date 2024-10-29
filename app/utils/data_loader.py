import csv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def truncateText(text, max_length=1000):
    return text[:max_length] if text else ""

def loadAndSplitData(file_paths):
    documents = []

    for file_path in file_paths:
        documents.extend(load_csv(file_path))

    logger.info(f"Número total de documentos cargados: {len(documents)}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    logger.info(f"Número de fragmentos de texto generados: {len(texts)}")
    if texts:
        logger.info(f"Muestra del primer fragmento:\n{texts[0].page_content[:500]}...")

    return texts

def load_csv(file_path):
    documents = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "syllabus" in file_path.lower():
                content = process_syllabus(row)
                doc_type = "syllabus"
            elif "promos" in file_path.lower():
                content = process_promo(row)
                doc_type = "promo"
            elif "deportes" in file_path.lower():
                content = process_deporte(row)
                doc_type = "deporte"
            elif "organizations" in file_path.lower():
                content = process_organizations(row)
                doc_type = "organization"
            elif "psychologist_faq" in file_path.lower():
                content = process_psychologist_faq(row)
                doc_type = "psychologist_faq"
            elif "psychologist_names" in file_path.lower():
                content = process_psychologist(row)
                doc_type = "psychologist_names"
            elif "empleos" in file_path.lower():
                content = process_empleos(row)
                doc_type = "empleos"
            else:
                logger.warning(f"Tipo de archivo desconocido: {file_path}")
                continue

            logger.debug(f"Procesando documento de tipo {doc_type}")
            logger.debug(f"Contenido: {content[:200]}...")  # Log the first 200 characters

            documents.append(Document(page_content=content, metadata={"source": file_path, "type": doc_type}))

    logger.info(f"Cargados {len(documents)} documentos desde {file_path}")
    return documents

def process_syllabus(row):
    # Asumiendo que el procesamiento de sílabos se mantiene igual
    return "\n".join([f"{k}: {truncateText(v)}" for k, v in row.items()])

def process_promo(row):
    return f"Lugar: {row.get('Lugar', 'No especificado')}\n" \
        f"Título: {row.get('Titulo', 'No especificado')}\n" \
        f"Descripción: {truncateText(row.get('Descripción', ''))}"

def process_deporte(row):
    return f"Categoría: {row.get('Categoría', 'No especificada')}\n" \
        f"Deporte: {row.get('Deporte', 'No especificado')}\n" \
        f"Tiempo de reserva: {row.get('Tiempo de reserva', 'No especificado')}\n" \
        f"Lugar: {row.get('Lugar', 'No especificado')}\n" \
        f"Link para reserva: {row.get('Link para hacer reserva', 'No especificado')}"

def process_organizations(row):
    return f"Tipo de organización: {row.get('Tipo de Organizacion', 'No especificado')}\n" \
        f"Nombre de organización: {row.get('Nombre de Organizacion', 'No especificado')}\n" \
        f"Correo de organización: {row.get('Correo de Organizacion', 'No especificado')}\n" \
        f"Descripción de la organización: {truncateText(row.get('Descripcion de la Organizacion', ''))}"

def process_psychologist_faq(row):
    return f"Pregunta: {row.get('pregunta', 'No especificada')}\n" \
        f"Respuesta: {row.get('respuesta', 'No especificada')}\n" \
        f"Categoria: {row.get('categoria', 'No especificada')}"

def process_psychologist(row):
    return f"Nombre: {row.get('Nombre', 'No especificado')}\n" \
        f"Especialidad: {row.get('Especialidad', 'No especificado')}\n" \
        f"Temas: {row.get('Temas', 'No especificado')}\n" \
        f"Modalidad_presencial: {row.get('Modalidad_presencial', 'No especificado')}\n" \
        f"Modalidad_virtual: {row.get('Modalidad_virtual', 'No especificado')}\n" \
        f"Link_de_cita: {row.get('Link_cita', 'No especificado')}"

def process_empleos(row):
    return f"Tipo_de_carrera: {row.get('carrera', 'No especificado')}\n" \
        f"Empresa: {row.get('empresa', 'No especificado')}\n" \
        f"Fecha_de_publicacion: {row.get('fecha_de_publicacion', 'No especificado')}\n" \
        f"Experiencia: {row.get('experiencia', 'No especificado')}\n" \
        f"Ingles_requerido: {row.get('ingles_requerido', 'No especificado')}"
