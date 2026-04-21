# AI Voice Assistant

A local AI voice assistant with:
- Face recognition (InsightFace buffalo_l, GPU)
- Wake word detection (openWakeWord - "Hey Jarvis")
- Speech to text (faster-whisper medium, GPU)
- LLM responses (Ollama qwen2.5:7b, local)
- Text to speech (Kokoro af_bella)
- Web search (Serper API)

## Setup

### 1. Clone the repo
git clone https://github.com/ReXoN10/ai-voice-assistant.git
cd ai-voice-assistant

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Install Ollama and pull model
ollama pull qwen2.5:7b

### 5. Add your face photos
Create a folder: face_data/YourName/
Add 5-10 clear photos of your face inside it.

### 6. Run
python pipeline.py

## Controls
- Say *"Hey Jarvis"* to ask a question
- Press *n* for latest news
- Press *w* for weather in Ahmedabad
- Press *q* to quit