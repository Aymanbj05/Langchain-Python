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


def start():
    print("Welcome to the LangChain Chatbot!")
    print("Type 'exit' to quit the program.")
    print(Fore.BLUE + "Bot: Hello! I am a chatbot. Ask me anything!" + fore.RESET)

    print("[1]-Raconter une blague")
    print("[2]-Quitter")
    choice= input("selectionner une option: ")
    if choice == "1":
        ask()
    elif choice == "2":
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    """Poser une question à l'IA"""
    while True:
        user_input = input("Topic: ")

        if user_input == "exit":
            start()
        else:
            response = chain.invoke({"topic": user_input})
            print(response)

if __name__ == "__main__":
    start() 