# Instalacion de libreria de OPENAI
!pip install openai
import openai

# Configura tu clave de API
openai.api_key = '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'

#Creamos un prompt
message_user = input("Ingresa tu pregunta para el BOT: ")

# Realiza una llamada a la API de OpenAI
response = openai.Completion.create(
  engine ="text-davinci-003",  # Elige el motor adecuado
  prompt = message_user,
  max_tokens = 1000  # cantidad de palabras en la respuesta
)

# Imprime la respuesta generada por GPT-3
answer_bot = response.choices[0].text.strip()
print("Respuesta del BOT: ", answer_bot)