# Import os to set and use our API key
import os
# Import OpenAI as our LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Let's use streamlit for our UI/app interface
import streamlit as st

# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader

# Import chroma to use it as our vector store
from langchain.vectorstores import Chroma

# Import vector store elements
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service we can also use other LLM providers

os.environ["OPENAI_API_KEY"] = "Your API KEY HERE"

# Create instance of OpenAI LLM

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load our PDF document
loader = PyPDFLoader('/Users/fatihbicer/PycharmProjects/FinanceGPT/Financial_Risk_Assessment.pdf')

# Let's split pages from our pdf
pages = loader.load_and_split()

# Load our document into vector database using ChromeDB
store = Chroma.from_documents(pages, embeddings, collection_name='capitalallocation')

# Let's create a vectorstore object
vectorstore_info = VectorStoreInfo(
    name="capital_allocation",
    description="an article about capital allocation",
    vectorstore=store
)

# Convert our vectorstore_info document into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end langchain with agent.
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Create a text input box for the user to enter a prompt
st.markdown ('<h4> BICER AI LLM </h4>', unsafe_allow_html = True)
st.title('Financial Risk Analysis GPT')
prompt = st.text_input('Input your prompt:')

# User to enter
if prompt:
    # Pass the prompt to the LLM

    response = agent_executor.run(prompt)

    # Write the response to the screen

    st.write(response)
# Answer from expander
    with st.expander('Document'):
        # Search for the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Write out the first related line
        st.write(search[0][0].page_content)
