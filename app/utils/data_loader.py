from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def truncate_text(text, max_length=1000):
    return text[:max_length]

def load_and_split_data(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    
    # Truncar y procesar cada documento
    processed_docs = []
    for doc in documents:
        # Buscar la clave del curso de forma case-insensitive
        curso_key = next((k for k in doc.metadata.keys() if k.lower() == 'curso'), None)
        contenido_key = next((k for k in doc.metadata.keys() if k.lower() == 'contenido'), None)
        
        if curso_key and contenido_key:
            content = f"Curso: {doc.metadata[curso_key]}\n{truncate_text(doc.metadata[contenido_key])}"
            processed_docs.append(Document(page_content=content, metadata=doc.metadata))
        else:
            print(f"Advertencia: Documento sin las columnas esperadas. Metadata: {doc.metadata}")
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(processed_docs)
    
    return texts
