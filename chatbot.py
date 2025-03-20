import speech_recognition as sr 
import pyttsx3  # text-to-speech module
from langchain_ollama import OllamaLLM  # to interact with OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
import whisper  # Whisper for offline speech-to-text transcription
import torch
import time 
import os 


engine = pyttsx3.init()

engine.setProperty('rate', 175) 
engine.setProperty('volume', 1)  
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0: male, 1: female


device = "cuda" if torch.cuda.is_available() else "cpu" 
model = whisper.load_model("small",device=device)  # You can use 'tiny', 'base', 'small'(Best), 'medium', or 'large'

# Function to speak text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize voice input using Whisper
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        wav_file_path = os.path.abspath(f"microphone_input.mp3")
        try:
            with open(wav_file_path, "wb") as f:
                f.write(audio.get_wav_data())

            wav_file_path = wav_file_path.replace("\\", "/")
            time.sleep(1)  # Delay for 1 second to ensure file is fully written
            print(f"Absolute path: {wav_file_path}")

            
            print("Recognizing audio...")
            result = model.transcribe(wav_file_path,language="english",fp16=False)
            print('Got result')
            text = result['text'].strip()
            print(f"\nYou: {text}")
            return text
        
        except FileNotFoundError as fnf_error:
            print(f"FileNotFoundError: {fnf_error}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File still exists: {os.path.exists(wav_file_path)}")
            speak_text("Sorry, there was an issue with the audio file. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")
            speak_text("An error occurred during speech recognition. Please try again.")
        finally:
            # Deleting the audio file
            try:
                os.unlink(wav_file_path)
                print(f"File removed: {wav_file_path}")
            except Exception as e:
                print(f"Error removing File: {e}")
    
    return ""



template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""


model_ollama = OllamaLLM(model="gemma2:2b")    # here using Gemma2 model, you can use your own
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model_ollama


def get_model_response(chain, context, user_input):
        try:
            result = chain.invoke({"context": context, "question": user_input})
            
            if isinstance(result, dict) and 'text' in result: 
                response_text = result['text']
            else:
                response_text = str(result)

            return response_text
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An unexpected error occurred. Please try again later."



def handle_conversation():
    context = ""
    print("\nWelcome to the world of OLLAMA 🙏")
    speak_text("Hello everyone, how are you all?")
    
    while True:
        user_input = recognize_speech()
        if not user_input or user_input=="." or user_input.lower() == "exit.":
            speak_text("Nice talking to you! Have a good day.")
            break
        
        response_text = get_model_response(chain, context, user_input)
        

        print("Ollama: ", response_text)
        speak_text(response_text)

        escape = ["good night.", "good bye.", "goodbye."]
        if user_input.lower() in escape:
            break

        # Update conversation history
        context += f"\nUser: {user_input}\nAI: {response_text}"

if __name__ == "__main__":
    handle_conversation()
