import os
from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough , RunnableMap

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

# 1. carregando os dados
loader = CSVLoader(file_path="dadosbasev1.csv" , encoding="utf-8")
documents = loader.load()
print(f"Total de documentos: {len(documents)}")


embeddings = GoogleGenerativeAIEmbeddings( google_api_key=os.getenv("GOOGLE_API_KEY") , model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embeddings)
retireval = vectorstore.as_retriever()

print(f"embedding ok")
# 2.1 Inicializando o modelo de linguagem Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

print(f"gemini ok")

rag_templalte = """
voce é um assistente pessoal de um cliente de uma empresa de tecnologia.
você deve responder as perguntas do cliente com base nos dados que você tem no contexto.

Contexto:
{context}

Pergunta:
{question}
"""

print(f"prompt ok")

prompt = ChatPromptTemplate.from_template(rag_templalte)
chain = (
    RunnableMap({ "context": retireval, "question": RunnablePassthrough() })
    | prompt
    | llm
)
print(f"chain ok")

# chain.invoke({"question": "Qual é o seu hobby?"})
# Sua pergunta

'''

minha_pergunta = "Qual seu animal favorito ?"

# Invocando a chain com a pergunta
resposta = chain.invoke(minha_pergunta)

# Exibindo a resposta
print(f"\nPergunta: {minha_pergunta}")
print(f"Resposta: {resposta.content}")

'''
# Vamos conversar com o modelo
def conversar_com_modelo(pergunta):
    print("Bem-vindo ao seu assistente vamos Conversando com o modelo... digite 'sair' para encerrar.\n")
    while True:
        # Solicita pergunta ao usuário
        pergunta = input("Digite sua pergunta: ")
        if pergunta.lower() == 'sair':
            print("Encerrando a conversa.")
            break
        resposta = chain.invoke(pergunta)
        # print(f"\nPergunta: {pergunta}")
        print(f"Resposta: {resposta.content}")

        # Pergunta ao usuário se deseja continuar
        continuar = input("Deseja fazer outra pergunta? (s/n): ").strip().lower()
        if continuar != 's':
            break

        # Solicita nova pergunta
        pergunta = input("Digite sua pergunta: ")

if __name__ == "__main__":
    conversar_com_modelo("Qual sua pergunta ?")
