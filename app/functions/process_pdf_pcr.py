import os
import re
import csv
import logging
from typing import List, Dict
from pdfminer.high_level import extract_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SilaboExtractor:
    def __init__(self, pdf_directory: str, output_file: str):
        self.pdf_directory = pdf_directory
        self.output_file = output_file

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[.\n]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_field(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        else:
            logging.warning(f"No se encontró coincidencia para el patrón '{pattern}'")
            return "No encontrado"

    def extract_info(self, text: str) -> Dict[str, str]:
        info = {}

        campos = [
            ("Carrera", r"(?:CARRERA|DEPARTAMENTO|DIRECCIÓN)\s*:?\s*(.*?)(?:\n|$)"),
            ("Curso", r"(?:CURSO|ASIGNATURA)\s*:?\s*(.*?)(?:\n|$)"),
            ("Malla", r"(?:MALLA|AÑO)\s*:?\s*(.*?)(?:\n|$)"),
            ("Modalidad", r"(?:MODALIDAD|2\.7\s*Modalidad:)\s*:?\s*(.*?)(?:\n|$)"),
            ("Creditos", r"(?:CREDITOS|CRÉDITOS|2\.2\s*Créditos:)\s*:?\s*(.*?)(?:\n|$)"),
        ]

        for nombre, patron in campos:
            info[nombre] = self.extract_field(text, patron)

        objetivos = re.findall(r"(?:Sesión|Objetivo)\s*\d*\s*:?\s*(.*?)(?:\n|$)", text)
        info['Objetivos'] = '; '.join(objetivos) if objetivos else self.extract_field(text, r"(?:OBJETIVOS|4\.\s*OBJETIVOS)(.*?)(?:\d+\.\s*COMPETENCIAS|\Z)")

        info['Competencias'] = self.extract_field(text, r"(?:COMPETENCIAS[^:]*:|5\.\s*COMPETENCIAS)(.*?)(?:\d+\.\s*RESULTADOS|\Z)")
        info['Resultados de Aprendizaje'] = self.extract_field(text, r"(?:RESULTADOS DE APRENDIZAJE|6\.\s*RESULTADOS)(.*?)(?:\d+\.\s*TEMAS|\Z)")
        info['Temas'] = self.extract_field(text, r"(?:TEMAS|7\.\s*TEMAS)(.*?)(?:\d+\.\s*PLAN|\Z)")
        info['Sistema de Evaluación'] = self.extract_field(text, r"(?:SISTEMA DE EVALUACIÓN|9\.\s*SISTEMA)(.*?)(?:\d+\.\s*REFERENCIAS|\Z)")
        info['Referencias Bibliográficas'] = self.extract_field(text, r"(?:REFERENCIAS BIBLIOGRÁFICAS|10\.\s*REFERENCIAS)(.*?)(?:\Z)")

        return info

    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        try:
            text = extract_text(pdf_path)
            return self.extract_info(text)
        except Exception as e:
            logging.error(f"Error procesando {pdf_path}: {e}")
            return {}

    def process_directory(self) -> List[Dict[str, str]]:
        results = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                logging.info(f"Procesando: {pdf_path}")
                result = self.process_pdf(pdf_path)
                if result:
                    result['Archivo'] = filename
                    results.append(result)
        return results

    def save_to_csv(self, data: List[Dict[str, str]]):
        if not data:
            logging.warning("No hay datos para guardar en el CSV")
            return

        fieldnames = ['Archivo', 'Carrera', 'Curso', 'Malla', 'Modalidad', 'Creditos', 
                      'Objetivos', 'Competencias', 'Resultados de Aprendizaje', 
                      'Temas', 'Sistema de Evaluación', 'Referencias Bibliográficas']

        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

        logging.info(f"Datos guardados en {self.output_file}")

    def run(self):
        logging.info("Iniciando procesamiento de sílabos")
        data = self.process_directory()
        self.save_to_csv(data)
        logging.info("Proceso finalizado")

if __name__ == "__main__":
    extractor = SilaboExtractor("./app/data/raw/syllabus_pdfs", "./app/data/syllabus_extracted.csv")
    extractor.run()
