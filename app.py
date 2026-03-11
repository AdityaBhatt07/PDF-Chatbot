from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        retriever = knowledge_base.as_retriever()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer the question using only the context below.
If you don't know the answer, just say you don't know.

Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        st.session_state.rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ PDF processed! Ask your question below.")

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        if st.session_state.rag_chain is None:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })

            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response))

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🧑 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 AI:** {msg.content}")


if __name__ == '__main__':
    main()
