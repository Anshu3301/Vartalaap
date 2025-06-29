# Vartalaap
This is a Python-based intelligent voice assistant that listens to your speech, understands your intent using the locally hosted **Gemma2 LLM** via **Ollama**, and responds back with natural-sounding speech. It’s fully offline-capable (except for dependencies) and uses **Whisper** for speech-to-text and **pyttsx3** for text-to-speech conversion.

Great for:
- Conversational AI experimentation
- Building personalized assistants
- Running completely offline (no APIs required)
- Educational and learning purposes

---

## ✨ Features

- 🎙️ **Voice Input**: Speak directly into your mic
- 🧠 **LLM-Powered Intelligence**: Uses `gemma2:2b` via Ollama
- 🔊 **Voice Output**: AI speaks back using `pyttsx3`
- 🔁 **Conversational Memory**: Maintains dialogue history for context
- ⚡ **Fast & Lightweight**: Uses `whisper-small` for speed & accuracy
- ❌ **No External API Needed**: Works without internet once set up

---

## Setup
1. Clone the repo.
2. Install Ollama (https://ollama.com/) on your machine and pull model: `ollama pull gemma2:2b`
   or any other model you wanna use.
3. Open terminal and go to specific folder,then run: `pip install ffmpeg torch whisper pyttsx3 SpeechRecognition langchain langchain_ollama`
4. Run the file(chatbot.py)


