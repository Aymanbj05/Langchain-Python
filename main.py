import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)

from langchain.prompts.chat import(
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import chroma
from colorama import Fore




# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

OPENAI_API_KEY= os.getenv["OPENAI_API_KEY"]
LANGUAGE_MODEL= "gpt-3.5-turbo"

template: str = """/
Vous êtes un spécialiste du support client. Vous assistez les utilisateurs avec des demandes générales basées sur {context}.
Si vous ne connaissez pas la réponse, vous invitez l'utilisateur à joindre le service support au téléphone ou par mail.
"""

# Create system and human message prompt templates
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="Question: {question}"
)

# Create the chat prompt template from the message templates
chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt, human_message_prompt
])

# Initialize the ChatOpenAI model
model = ChatOpenAI(api_key=OPENAI_API_KEY)


# Placeholder function for loading documents
def load_documents():
    """
    Load a file from path, split it into chunks, embed each chunk, and load it into the vector store.
    """
    loader= TextLoader(".txt")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    documents= loader.load()
    chunks=text_splitter.split_documents(documents)
    print(f"You have {len(documents)} documents and {len(chunks)}")
    return chunks

# Placeholder function for loading embeddings
def load_embeddings(documents, user_query):
    """
    Create a vector store for the given document and user query.
    """
    db= chroma.from_documents(documents, OpenAIEmbeddings)
    docs= db.similarity_search(user_query)
    print(docs)
    return db.as_retrievar()
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
    print(Fore.BLUE + "Bot: Hello! I am a chatbot. Ask me anything!" + Fore.RESET)

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


def generate_response(retriever, context):
    """
    Generate a response based on the provided question and context using the language model.
    """
    chain= (
        {context: retriever , "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()

    )
    return chain.invoke(query)
    


def query(query):
    documents= load_documents()
    retriever = load_embeddings(documents, query)
    response= generate_response(retriever, query)
    print(Fore.GREEN + response)

query("Quelle sont les horaires d'ouverture?")


def start():
    print("Welcome to the LangChain Chatbot!")
    print("Type 'exit' to quit the program.")
    print(Fore.BLUE + "Bot: Hello! I am a chatbot. Ask me anything!" + Fore.RESET)

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
            response = query(user_input)
            print(response)

if __name__ == "__main__":
    start() 
    


  