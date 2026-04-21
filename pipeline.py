from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import ollama
import requests
import os
import threading
import time
import queue
import warnings
import sounddevice as sd
import pickle

from kokoro import KPipeline
from insightface.app import FaceAnalysis
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR    = "face_data"
OLLAMA_MODEL       = "qwen2.5:7b"
SERPER_KEY         = os.getenv("SERPER_KEY")
FACE_INTERVAL      = 20
TTS_VOICE          = "af_aoede"
TTS_SPEED          = 1.0
SIM_THRESHOLD      = 0.35
WHISPER_MODEL_SIZE = "medium"
WHISPER_LANGUAGE   = "en"
SILENCE_THRESHOLD  = 150        # RMS units out of 32768
SILENCE_DURATION   = 1.2        # seconds of silence → stop recording
MAX_RECORD_SECONDS = 10         # hard cap
WAKE_WORD_DELAY    = 1.2        # seconds after "Yes?" before mic opens
MAX_DISPLAY_LINES  = 6          # max lines shown in the CV2 response bar

# ─── Long-response detection keywords ────────────────────────────────────────
LONG_RESPONSE_KEYWORDS = [
    "essay", "explain", "describe", "paragraph", "words",
    "write", "tell me about", "summarize", "detail", "elaborate",
    "how does", "what is", "history of", "story", "speech",
]

HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching",
    "subscribe", "like and subscribe",
]


# ══════════════════════════════════════════════════════════════════════════════
#  STATE MACHINE
#  One state at a time. All threads read it; only orchestrator writes it.
# ══════════════════════════════════════════════════════════════════════════════
class State:
    IDLE         = "IDLE"
    LISTENING    = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING     = "THINKING"
    SPEAKING     = "SPEAKING"

current_state = State.IDLE
state_lock    = threading.Lock()

def set_state(s):
    global current_state
    with state_lock:
        print(f"  ── {current_state} → {s}")
        current_state = s

def get_state():
    with state_lock:
        return current_state


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DISPLAY DATA
# ══════════════════════════════════════════════════════════════════════════════
data = {
    "person_name":   "No Face",
    "face_box":      None,
    "face_color":    (180, 180, 180),
    "face_label":    "Scanning...",
    "last_response": "",
    "search_status": "",
    "greeted":       set(),
}


# ══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNIZER
# ══════════════════════════════════════════════════════════════════════════════
class FaceRecognizer:
    def __init__(self):
        print("Loading InsightFace...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.threshold = SIM_THRESHOLD
        self.known_embs, self.known_names = self._load(KNOWN_FACES_DIR)
        print(f"InsightFace ready — {len(self.known_names)} face(s) enrolled")

    def _load(self, face_dir):
        cache = os.path.join(face_dir, "_cache.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                d = pickle.load(f)
            print(f"  Face cache loaded ({len(d[1])} encoding(s))")
            return d
        print("  Building face cache...")
        embs, names = [], []
        for person in os.listdir(face_dir):
            pdir = os.path.join(face_dir, person)
            if not os.path.isdir(pdir):
                continue
            for img_file in os.listdir(pdir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(pdir, img_file))
                if img is None:
                    continue
                faces = self.app.get(img)
                if faces:
                    embs.append(faces[0].normed_embedding)
                    names.append(person)
            print(f"  ✔ {person}")
        with open(cache, "wb") as f:
            pickle.dump((embs, names), f)
        return embs, names

    def recognize(self, bgr_frame):
        faces = self.app.get(bgr_frame)
        if not faces:
            return "No Face", 0.0, None
        face = max(faces, key=lambda f: f.det_score)
        box  = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        if not self.known_embs:
            return "Stranger", float(face.det_score), (x1, y1, x2, y2)
        sims     = np.dot(self.known_embs, face.normed_embedding)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self.threshold:
            return self.known_names[best_idx], best_sim, (x1, y1, x2, y2)
        return "Stranger", best_sim, (x1, y1, x2, y2)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO OUTPUT  (always-on callback-driven stream)
# ══════════════════════════════════════════════════════════════════════════════
audio_out_q = queue.Queue()
_leftover   = np.array([], dtype="float32")

def _out_callback(outdata, frames, time_info, status):
    global _leftover
    out  = np.zeros(frames, dtype="float32")
    pos  = 0
    need = frames
    if len(_leftover):
        take = min(len(_leftover), need)
        out[pos:pos+take] = _leftover[:take]
        _leftover = _leftover[take:]
        pos += take; need -= take
    while need > 0:
        try:
            chunk = audio_out_q.get_nowait()
            if chunk is None:
                audio_out_q.task_done(); break
            chunk = np.asarray(chunk, dtype="float32").flatten()
            take  = min(len(chunk), need)
            out[pos:pos+take] = chunk[:take]
            _leftover = chunk[take:]
            pos += take; need -= take
            audio_out_q.task_done()
        except queue.Empty:
            break
    outdata[:, 0] = out

def _clear_audio():
    global _leftover
    _leftover = np.array([], dtype="float32")
    while not audio_out_q.empty():
        try:
            audio_out_q.get_nowait()
            audio_out_q.task_done()
        except queue.Empty:
            break

def _audio_playing():
    return not audio_out_q.empty() or len(_leftover) > 0

out_stream = sd.OutputStream(
    samplerate=24000, channels=1, dtype="float32",
    blocksize=2048, callback=_out_callback,
)
out_stream.start()


# ══════════════════════════════════════════════════════════════════════════════
#  TTS  — fully blocking (returns only when playback is complete)
# ══════════════════════════════════════════════════════════════════════════════
def speak_and_wait(text, tts_pipeline):
    if not text:
        return
    _clear_audio()
    try:
        for _, _, audio in tts_pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED):
            audio_out_q.put(np.asarray(audio, dtype="float32").flatten())
    except Exception as e:
        print(f"TTS error: {e}")
        return
    while _audio_playing():
        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def web_search(query):
    try:
        data["search_status"] = "Searching..."
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 3},
            timeout=5,
        )
        results = r.json().get("organic", [])
        data["search_status"] = ""
        return "\n\n".join(
            f"Title: {x.get('title','')}\nSummary: {x.get('snippet','')}"
            for x in results
        )
    except Exception as e:
        data["search_status"] = ""
        print(f"Search error: {e}")
        return ""

def needs_search(question):
    try:
        prompt = (
    f'You are deciding whether to search the web before answering.\n'
    f'Reply YES if the question is about ANY of these: '
    f'sports scores, match results, cricket, IPL, news, accidents, events, '
    f'weather, prices, stocks, current affairs, or anything that happened recently.\n'
    f'Reply NO only if the question is purely factual/educational '
    f'(science, history, math, definitions).\n'
    f'Question: "{question}"\n'
    f'Reply with only YES or NO.'
)
        r = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 5},
        )
        answer = r["message"]["content"].strip().upper()
        print(f"  Search needed? {answer}")
        return "YES" in answer
    except Exception as e:
        print(f"  needs_search error: {e}")
        FALLBACK = [
            "news", "today", "latest", "current", "now", "weather", "price",
            "score", "update", "recent", "what happened", "who won", "stock",
        ]
        return any(t in question.lower() for t in FALLBACK)


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA  — synchronous, returns full reply string
# ══════════════════════════════════════════════════════════════════════════════
def ask_ollama(name, question):
    display = name if name not in ("No Face", "Stranger", "Unknown") else "there"

    # Dynamic token limit — long if essay/explanation requested
    wants_long = any(k in question.lower() for k in LONG_RESPONSE_KEYWORDS)
    max_tokens = 600 if wants_long else 150
    print(f"  Token limit: {max_tokens} ({'long' if wants_long else 'short'} response)")

    if needs_search(question):
        print(f"  Web search: {question}")
        results = web_search(question)
        if results:
            prompt = (
                f"You are a friendly AI assistant talking to {display}.\n"
                f'They asked: "{question}"\n\n'
                f"Fresh web search results:\n{results}\n\n"
                f"Answer in a natural conversational way using the results. "
                f"Address {display} by name."
            )
        else:
            prompt = (
                f"You are a friendly AI assistant talking to {display}.\n"
                f'They asked: "{question}"\n'
                f"Answer in a natural conversational way. Address {display} by name."
            )
    else:
        prompt = (
            f"You are a friendly AI assistant talking to {display}.\n"
            f'They asked: "{question}"\n'
            f"Answer in a natural conversational way. Address {display} by name."
        )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": max_tokens},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return "Sorry, I had trouble thinking of a response."

def greet_ollama(name):
    display = name if name not in ("No Face", "Stranger", "Unknown") else "there"
    if name not in ("No Face", "Stranger", "Unknown"):
        prompt = (
            f"You are a friendly AI assistant.\n"
            f"{display} just appeared in front of the camera.\n"
            f"Give ONE short warm greeting to {display}."
            f"Under 12 words. No questions."
        )
    else:
        prompt = (
            "You are a friendly AI assistant.\n"
            "An unknown person appeared. Give ONE short friendly greeting. Under 12 words."
        )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8, "num_predict": 40},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama greet error: {e}")
        return f"Hey {display}, good to see you!"


# ══════════════════════════════════════════════════════════════════════════════
#  WHISPER TRANSCRIPTION  — synchronous
# ══════════════════════════════════════════════════════════════════════════════
def transcribe(audio_buffer, whisper_model):
    audio_np = np.array(audio_buffer, dtype="float32")

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    if rms < (80 / 32768.0):
        print("  (too quiet — skipped)")
        return ""

    if len(audio_buffer) < 16000 * 1.0:
        print("  (too short — skipped)")
        return ""

    try:
        segments, _ = whisper_model.transcribe(
            audio_np,
            language=WHISPER_LANGUAGE,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"Whisper error: {e}")
        return ""

    if not text:
        return ""

    tl = text.lower()
    if any(p in tl for p in HALLUCINATION_PHRASES):
        print(f"  (hallucination filtered: '{text}')")
        return ""

    if len(text.split()) < 2:
        print(f"  (too short to be real: '{text}')")
        return ""

    return text


# ══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION WORKER  (background thread)
# ══════════════════════════════════════════════════════════════════════════════
face_in_q  = queue.Queue(maxsize=1)
face_out_q = queue.Queue(maxsize=1)

def face_worker(recognizer):
    while True:
        try:
            frame = face_in_q.get(timeout=1)
        except queue.Empty:
            continue
        name, conf, box = recognizer.recognize(frame)
        color = (0, 220, 0) if name not in ("No Face", "Stranger", "Unknown") else (0, 0, 220)
        label = f"{name} ({conf:.0%})" if box is not None else "No face detected"
        result = {"name": name, "box": box, "color": color, "label": label}
        if not face_out_q.full():
            face_out_q.put(result)
        face_in_q.task_done()


# ══════════════════════════════════════════════════════════════════════════════
#  WAKE WORD LISTENER  (background thread)
# ══════════════════════════════════════════════════════════════════════════════
wake_event = threading.Event()

def wake_word_listener():
    print("Loading wake word model...")
    wake_model = WakeWordModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    print("Wake word ready — say 'Hey Jarvis'")

    def callback(indata, frames, time_info, status):
        if get_state() != State.IDLE:
            return
        pcm_int16 = (indata[:, 0] * 32767).astype(np.int16)
        score = wake_model.predict(pcm_int16).get("hey_jarvis", 0.0)
        if score > 0.5:
            wake_event.set()

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=1280, callback=callback):
        while True:
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════════════
#  MIC RECORDER  — blocking, returns audio buffer when done
# ══════════════════════════════════════════════════════════════════════════════
def record_until_silence():
    SAMPLE_RATE   = 16000
    buf           = []
    silence_start = None
    done          = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal silence_start
        if get_state() != State.LISTENING:
            done.set()
            return
        pcm = indata[:, 0]
        buf.extend(pcm.tolist())
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        if rms < (SILENCE_THRESHOLD / 32768.0):
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                done.set()
        else:
            silence_start = None
        if len(buf) >= SAMPLE_RATE * MAX_RECORD_SECONDS:
            done.set()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=1280, callback=callback):
        done.wait()

    return buf


# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD QUESTION INJECTOR
# ══════════════════════════════════════════════════════════════════════════════
def _inject_question(question):
    while get_state() != State.IDLE:
        time.sleep(0.1)
    print(f"\n[Injected question]: {question}")
    _run_response_pipeline(question)

def _run_response_pipeline(question):
    set_state(State.THINKING)
    name  = data["person_name"]
    reply = ask_ollama(name, question)
    print(f"[Jarvis]: {reply}")
    data["last_response"] = reply
    set_state(State.SPEAKING)
    speak_and_wait(reply, tts_pipeline)
    set_state(State.IDLE)


# ══════════════════════════════════════════════════════════════════════════════
#  INIT ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Initializing Jarvis...")

recognizer = FaceRecognizer()

print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
print("Whisper ready!")

print("Loading Kokoro TTS...")
tts_pipeline = KPipeline(lang_code="a")
try:
    for _ in tts_pipeline("Hello.", voice=TTS_VOICE, speed=TTS_SPEED):
        break
    print("TTS ready!")
except Exception as e:
    print(f"TTS warmup warning: {e}")

print(f"Ollama model: {OLLAMA_MODEL}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  START BACKGROUND THREADS
# ══════════════════════════════════════════════════════════════════════════════
threading.Thread(target=face_worker,        args=(recognizer,), daemon=True).start()
threading.Thread(target=wake_word_listener,                     daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def orchestrator():
    while True:
        if get_state() != State.IDLE:
            time.sleep(0.05)
            continue

        woke = wake_event.wait(timeout=0.2)
        if not woke:
            continue

        wake_event.clear()
        print("\n[Wake word detected]")

        set_state(State.SPEAKING)
        speak_and_wait("Yes?", tts_pipeline)
        time.sleep(WAKE_WORD_DELAY)

        set_state(State.LISTENING)
        print("[Listening — speak now]")
        audio_buf = record_until_silence()
        print(f"[Done — {len(audio_buf)/16000:.1f}s captured]")

        set_state(State.TRANSCRIBING)
        text = transcribe(audio_buf, whisper_model)

        if not text:
            print("[Nothing usable — back to IDLE]")
            speak_and_wait(
                "Sorry, I didn't catch that. Say Hey Jarvis to try again.",
                tts_pipeline
            )
            set_state(State.IDLE)
            continue

        print(f"[You said]: {text}")
        _run_response_pipeline(text)


def face_greeter():
    while True:
        time.sleep(1)
        if get_state() != State.IDLE or _audio_playing():
            continue
        name = data["person_name"]
        if name in ("No Face", "Stranger", "Unknown"):
            continue
        if name in data["greeted"]:
            continue
        data["greeted"].add(name)
        print(f"\n[Greeting {name}]")
        set_state(State.THINKING)
        greeting = greet_ollama(name)
        print(f"[Jarvis]: {greeting}")
        data["last_response"] = greeting
        set_state(State.SPEAKING)
        speak_and_wait(greeting, tts_pipeline)
        set_state(State.IDLE)


threading.Thread(target=orchestrator,  daemon=True).start()
threading.Thread(target=face_greeter,  daemon=True).start()
print("Jarvis is running. Say 'Hey Jarvis' to begin.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LOOP  (main thread)
# ══════════════════════════════════════════════════════════════════════════════
cap         = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break

    frame_count += 1

    if frame_count % FACE_INTERVAL == 0 and not face_in_q.full():
        face_in_q.put(frame.copy())

    try:
        result = face_out_q.get_nowait()
        data["person_name"] = result["name"]
        data["face_box"]    = result["box"]
        data["face_color"]  = result["color"]
        data["face_label"]  = result["label"]
    except queue.Empty:
        pass

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n") and get_state() == State.IDLE:
        threading.Thread(
            target=_inject_question,
            args=("What are the top news headlines today?",),
            daemon=True,
        ).start()
    elif key == ord("w") and get_state() == State.IDLE:
        threading.Thread(
            target=_inject_question,
            args=("What is the weather in Ahmedabad India today?",),
            daemon=True,
        ).start()

    # Draw face box
    if data["face_box"] is not None:
        x1, y1, x2, y2 = data["face_box"]
        c = data["face_color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.rectangle(frame, (x1, y1-32), (x2, y1), c, -1)
        cv2.putText(frame, data["face_label"], (x1+5, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Draw last response (shows last MAX_DISPLAY_LINES lines only)
    if data["last_response"]:
        words       = data["last_response"].split()
        lines, line = [], ""
        for word in words:
            if len(line + word) < 55:
                line += word + " "
            else:
                lines.append(line.strip())
                line = word + " "
        lines.append(line.strip())
        lines   = [l for l in lines if l]
        lines   = lines[-MAX_DISPLAY_LINES:]    # ← only last N lines
        bar_h   = 30 + len(lines) * 25
        h       = frame.shape[0]
        cv2.rectangle(frame, (0, h - bar_h), (frame.shape[1], h), (0, 0, 0), -1)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (10, h - bar_h + 22 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Status bar
    status_map = {
        State.IDLE:         ("Say 'Hey Jarvis' to speak  |  n=news  w=weather", (180, 180, 180)),
        State.LISTENING:    ("Listening...",                                      (0, 255, 100)),
        State.TRANSCRIBING: ("Transcribing...",                                   (0, 200, 255)),
        State.THINKING:     ("Thinking...",                                       (200, 200, 0)),
        State.SPEAKING:     ("Speaking...",                                       (0, 255, 150)),
    }
    label, color = status_map.get(get_state(), ("...", (180, 180, 180)))
    if data["search_status"]:
        label, color = data["search_status"], (0, 165, 255)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imshow("Jarvis", frame)


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()
out_stream.stop()
out_stream.close()
print("Closed.")