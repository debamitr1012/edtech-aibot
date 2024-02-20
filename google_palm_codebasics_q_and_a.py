from langchain.llms import GooglePalm
api_key = 'GOOGLE_API_KEY' # get this free api key from https://makersuite.google.com/
llm = GooglePalm(google_api_key=api_key, temperature=0.1)
poem = llm("Write a 4 line poem of my love for samosa")
print(poem)
essay = llm("write email requesting refund for electronic item")
print(essay)
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='C:/Users/91983/Downloads/3_project_codebasics_q_and_a/codebasics_faqs.csv', source_column="prompt")
# Store the loaded data in the 'data' variable
data = loader.load()
from langchain.embeddings import HuggingFaceInstructEmbeddings
# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
e = instructor_embeddings.embed_query("What is your refund policy?")
len(e)
e[:5]
from langchain.vectorstores import FAISS
# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,
                                 embedding=instructor_embeddings)
# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold = 0.7)
rdocs = retriever.get_relevant_documents("how about job placement support?")
rdocs
# google_palm_embeddings = GooglePalmEmbeddings(google_api_key=api_key)
# from langchain.vectorstores import Chroma
# vectordb = Chroma.from_documents(data,
#                            embedding=google_palm_embeddings,
#                            persist_directory='./chromadb')
# vectordb.persist()
from langchain.prompts import PromptTemplate
prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
CONTEXT: {context}
QUESTION: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
chain('Do you provide job assistance and also do you provide job gurantee?')
chain("Do you guys provide internship and also do you offer EMI payments?")
chain("do you have javascript course?")
chain("Do you have plans to launch blockchain course in future?")
chain("should I learn power bi or tableau?")
chain("I've a MAC computer. Can I use powerbi on it?")
chain("I don't see power pivot. how can I enable it?")
chain("What is the price of your machine learning course?")