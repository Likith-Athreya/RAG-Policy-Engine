import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = DirectoryLoader("docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

prompt1 = ChatPromptTemplate.from_template(
    "Answer based on context: {context}\nQuestion: {input}"
)

system_prompt2 = (
    "You are a professional Policy Assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say 'I do not have sufficient information.' "
    "Format your answer using bullet points for clarity. Include section names if available."
    "\nContext: {context}"
)
prompt2 = ChatPromptTemplate.from_messages([
    ("system", system_prompt2),
    ("human", "{input}")
])

def run_rag(user_query, prompt_template):
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain.invoke({"input": user_query})

if __name__ == "__main__":
    test_questions = [
        "An American customer wants to return a laptop they bought 25 days ago. Based on the Global Returns and Refunds Policy, are they eligible for a return?",
        "What is the express shipping cost for a package weighing 15.0 lbs?",
        "Can a user get a refund for a digital software download after they have started the byte-stream download?",
        "Under what condition is the 5-7 day delivery guarantee suspended according to the 'Force Majeure' clause?",
        "What legal action does a user waive by agreeing to the 'Mandatory Arbitration' section of the Terms of Service?"
    ]

    for query in test_questions:
        print(f"\nQUERY: {query}")
        
        resp1 = run_rag(query, prompt1)
        resp2 = run_rag(query, prompt2)

        print(f"\n[PROMPT V1 ANSWER]:{resp1['answer']}")
        print(f"\n[PROMPT V2 ANSWER]:{resp2['answer']}")
        
        print("\n[SOURCES]:")
        for doc in resp2["context"]:
            print(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")