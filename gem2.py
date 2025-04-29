import os
from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough , RunnableMap

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

loader = CSVLoader(file_path="dadosbasev1.csv" , encoding="utf-8")
documents = loader.load()
print(f"Concatado com sucesso a base de dados {len(documents)} documentos.")


embeddings = GoogleGenerativeAIEmbeddings( google_api_key=os.getenv("GOOGLE_API_KEY") , model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embeddings)
retireval = vectorstore.as_retriever()


# 2.1 Inicializando o modelo de linguagem Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)



rag_templalte = """
voce é um assistente pessoal de um cliente de uma empresa de tecnologia.
você deve responder as perguntas do cliente com base nos dados que você tem no contexto.

Contexto:
{context}

Pergunta:
{question}
"""



prompt = ChatPromptTemplate.from_template(rag_templalte)
chain = (
    RunnableMap({ "context": retireval, "question": RunnablePassthrough() })
    | prompt
    | llm
)


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
        # pergunta = input("Digite sua pergunta: ")

if __name__ == "__main__":
    conversar_com_modelo("Qual sua pergunta ?")
