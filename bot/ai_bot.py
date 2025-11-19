# Neste arquivo está nossas configurações para o modelo de linguagem que escolhemos
# ESTE PROJETO FAZ UM TESTE COM GROQ
#     [ Altere aqui o prompt de comando da IA ]
#-----------------------------------------------------------------------------------#

# BIBLIOTECAS
import os

from decouple import config

# [ NOTAS IMPORTANTES ]
# NA LINHA 'from langchain.chains.combine_documents.stuff import create_stuff_documents_chain  # type: ignore'
# O PYLANCE NÃO RECONHECEU O MÓDULO DE PRIMEIRA E TEVE QUE FAZER UMA "GAMBIARRA"
# -> add o '.stuff' na linha do código, -> upgrade das bibliotecas 'langchain' e 'langchain-core'
# -> add um comentário para suprimir o 'warning' 

# Isso silencia o Pylance e mantém o código limpo.

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain  # type: ignore


from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["API_KEY_HERE"] = config("API_KEY_HERE")


class AIBot:

    def __init__(self):
        self.__chat = ChatGroq(model='llama-3.1-70b-versatile')
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={'k': 30},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            messages.append(message_class(content=message.get('body')))
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        SYSTEM_TEMPLATE = '''
        Responda as perguntas dos usuários com base no contexto abaixo.
        Você é um assistente especializado em tirar dúvidas sobre o treinamento Django Master da PycodeBR.
        Tire dúvidas dos possíveis alunos que entrarem em contato.
        Responda de forma natural, agradável e respeitosa. Seja objetivo nas respostas, com informações
        claras e diretas. Foque em ser natural e humanizado, como um diálogo comum entre duas pessoas.
        Leve em consideração também o histórico de mensagens da conversa com o usuário.
        Responda sempre em português brasileiro.

        <context>
        {context}
        </context>
        '''

        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages, question),
            }
        )
        return response
