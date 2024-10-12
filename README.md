<h1 align="center">MindTEC: Asistente Virtual UTEC</h1>

MindTEC es un asistente virtual inteligente diseñado específicamente para los estudiantes de la Universidad de Ingeniería y Tecnología (UTEC). Utilizando tecnologías de procesamiento de lenguaje natural avanzadas, MindTEC proporciona información precisa y relevante sobre sílabos de cursos, promociones universitarias, actividades deportivas y más.

## ✨ Características Principales
- 🎓 Información detallada sobre sílabos de cursos
- 🏷️ Promociones y descuentos para estudiantes
- 🏀 Reservas de instalaciones deportivas
- 💬 Interfaz de chat intuitiva vía WhatsApp

## 🛠️ Tecnologías Utilizadas
- Python 3.8+
- FastAPI
- LangChain
- Qdrant (Vector Database)
- OpenAI GPT-4
- Twilio (para integración con WhatsApp)

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Cuenta en Qdrant Cloud
- Cuenta en OpenAI
- Cuenta en Twilio

### Pasos de Instalación
1. Clona el repositorio:
   ```
   git clone https://github.com/kaloslazo/MindTEC.git
   cd mindtec
   ```

2. Crea y activa un entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno:
   Crea un archivo `.env` en la raíz del proyecto y añade las siguientes variables:
   ```
   OPENAI_API_KEY=tu_clave_api_de_openai
   QDRANT_URL=tu_url_de_qdrant
   QDRANT_API_KEY=tu_clave_api_de_qdrant
   QDRANT_COLLECTION_NAME=nombre_de_tu_coleccion
   TWILIO_ACCOUNT_SID=tu_sid_de_twilio
   TWILIO_AUTH_TOKEN=tu_token_de_autenticacion_de_twilio
   TWILIO_PHONE_NUMBER=tu_numero_de_whatsapp_de_twilio
   ```

5. Inicia la aplicación:
   ```
   uvicorn app.main:app --reload
   ```

6. Configura el webhook de Twilio:
    - Abre la consola de Twilio y navega a la configuración de tu número de WhatsApp.
    - En la sección "A Message Comes In", selecciona "Webhook" y añade la URL de tu servidor FastAPI.
    - Asegúrate de que el método HTTP sea "POST".

---

🚀 ¡Disfruta usando MindTEC y mejora tu experiencia universitaria en UTEC! 📚🎓

