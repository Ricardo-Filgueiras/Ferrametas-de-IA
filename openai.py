import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI



_ = load_dotenv()

# 1. carregando os dados
loader = CSVLoader(file_path="dadosbasev1.csv" , encoding="utf-8")
documents = loader.load()
print(f"Total de documentos: {len(documents)}")


# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retireval = vectorstore.as_retriever()

# 2 inicializando o modelo de linguagem OpenAI

llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
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

chain.invoke({"question": "Qual é o seu hobby?"})