from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from colorama import Fore
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def format_response(result):
    return f"{Fore.GREEN}{result}{Fore.RESET}"


# Créer une chaîne de traitement
prompt=ChatPromptTemplate.from_template("Tell me a short joke about {topic}")    

model= ChatOpenAI(model="gpt-3.5-turbo")

chain= prompt | model | format_response

chain.invoke({"topic":"chicken"})
