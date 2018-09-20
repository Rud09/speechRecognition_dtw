import speech_recognition as sr
r=sr.Recognizer()
harvard=sr.AudioFile('harvard.wav')
with harvard as source:
    audio = r.record()
r.recognize_google(audio)
