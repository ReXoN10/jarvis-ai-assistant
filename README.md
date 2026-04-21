# Jarvis — Local AI Voice Assistant

A fully local, real-time AI assistant that recognizes faces, listens for a wake word, understands speech, and responds with a natural voice. No cloud AI — everything runs on your machine.

## Features

- **Face recognition** — identifies known people via webcam and greets them by name (InsightFace)
- **Wake word detection** — activates on "Hey Jarvis" (OpenWakeWord)
- **Speech to text** — transcribes your question locally (faster-whisper medium, GPU)
- **Local LLM** — generates responses via Ollama running qwen2.5:7b on device
- **Text to speech** — speaks responses aloud (Kokoro TTS)
- **Web search** — fetches live results for news, weather, sports via Serper API
- **State machine architecture** — clean IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING flow managed across threads

## Tech Stack

| Component | Library |
|---|---|
| Face recognition | InsightFace (buffalo_l) |
| Wake word | OpenWakeWord |
| Speech to text | faster-whisper |
| LLM | Ollama (qwen2.5:7b) |
| Text to speech | Kokoro TTS |
| Web search | Serper API |
| Camera / display | OpenCV |
| Audio I/O | sounddevice |

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA recommended
- Ollama installed and running locally
- Serper API key (free tier available at serper.dev)

## Installation

```bash
git clone https://github.com/ReXoN10/jarvis-ai-assistant.git
cd jarvis-ai-assistant
python -m venv venv
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and add your Serper API key:

```
SERPER_KEY=your_key_here
```

## Add Your Face (Optional)

Create a folder with your name inside `face_data/` and add 2-3 clear photos:

```
face_data/
└── YourName/
    ├── photo1.jpg
    └── photo2.jpg
```

Jarvis will greet you by name when it sees your face.

## Run

```bash
ollama serve
python pipeline.py
```

Then say **"Hey Jarvis"** to activate.

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `n` | Ask for today's news headlines |
| `w` | Ask for current weather in Ahmedabad |
| `q` | Quit |

## How It Works

```
Wake word detected
        ↓
    Says "Yes?"
        ↓
  Records audio until silence
        ↓
  Transcribes with Whisper
        ↓
  Decides if web search needed
        ↓
  Generates response via Ollama
        ↓
  Speaks response via Kokoro TTS
        ↓
     Back to IDLE
```

## Known Limitations

- No conversation memory — each question is independent
- Global shared state dictionary is not fully thread-safe for concurrent writes
- Wake word occasionally triggers on similar-sounding phrases

## What I Would Improve Next

- Add per-person conversation history and memory
- Replace global data dict with proper thread-safe state management
- Add a proper GUI instead of OpenCV overlay