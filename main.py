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


    

