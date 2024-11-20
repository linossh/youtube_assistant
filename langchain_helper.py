from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_for_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(transcript)    

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(model="gpt-4o")
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {context}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be detailed.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, context=docs_page_content)
    response = response.replace(".", ".\n")
    return response, docs