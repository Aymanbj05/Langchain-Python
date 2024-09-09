from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from colorama import Fore
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API depuis l'environnement
api_key = os.getenv("OPENAI_API_KEY")

# Vérifier si la clé API est chargée correctement
if not api_key:
    raise ValueError("La clé API OpenAI n'est pas définie. Vérifiez votre fichier .env.")

# Afficher la clé API pour déboguer
print(f"Clé API chargée: {api_key[:5]}...")  # Affiche les 5 premiers caractères pour vérifier

# Initialiser le modèle avec la clé API
model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")

# Exemple d'utilisation du modèle
response = model.invoke("Tell me a short joke about horses")
print(response)