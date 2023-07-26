# Importing required libraries

import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import time
import pandas as pd
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import VertexAI
from langchain import PromptTemplate
import streamlit as st
from langchain import FAISS
from langchain.document_loaders.sitemap import SitemapLoader
from urllib.parse import urlparse
from config import *

# Constants for retrying and backoff
NUM_RETRIES = 10
BACKOFF_FACTOR = 1

# Function to validate the URL scheme and standardize the URL


def validate_url_scheme(url) -> str:
    # Check if URL starts with "http" or "https"
    if not url.startswith('http'):
        msg = "The URL must begin with http:// or https://"
        exit_code = 1
        url = ""
    else:
        msg = "URL Scheme Validated"
        exit_code = 0
        url = standardise_url(url)
    return (msg, exit_code, url)

# Function to standardize the URL with or without 'www'


def standardise_url(url):
    parsed_url = urlparse(url)
    final_url = 'www.' + \
        parsed_url.netloc if not parsed_url.netloc.startswith(
            'www.') else parsed_url.netloc
    return parsed_url.scheme + '://' + final_url + parsed_url.path

# Function to get the base URL from the provided URL


def get_base_url(url):
    parsed_url = urlparse(url)
    final_url = 'www.' + \
        parsed_url.netloc if not parsed_url.netloc.startswith(
            'www.') else parsed_url.netloc
    return final_url

# Caching decorator to optimize web crawling process


@st.cache_data(ttl=86400, show_spinner="Parsing the website...")
def get_docs(url):
    # Variables to track if all URLs are parsed
    all_urls_parsed = False
    final_docs = []
    error_urls = []
    # Initialize SitemapLoader with web path (URL)
    sitemap_loader = SitemapLoader(web_path=url)
    # Continue until all URLs are parsed
    while not all_urls_parsed:
        sitemap_loader.requests_per_second = 10
        sitemap_loader.raise_for_status = True
        # Retry the web request if there is an error
        for retry_num in range(NUM_RETRIES):
            time_to_sleep = BACKOFF_FACTOR * (2 ** (retry_num - 1))
            try:
                initial_docs = sitemap_loader.load()
                if len(initial_docs) != len(error_urls) and len(error_urls) > 1:
                    raise requests.exceptions.HTTPError
                break
            except requests.exceptions.HTTPError:
                # Retry with exponential backoff if there is an error
                time.sleep(time_to_sleep)
                pass

        # Filter out documents with error content and append to the final list
        final_docs.extend([doc for doc in initial_docs if doc.page_content !=
                          '429 Too Many Requests\nYou have sent too many requests in a given amount of time.\n\n'])
        # Update error URLs for retry
        error_urls = [doc.metadata['source'] for doc in initial_docs if doc.page_content ==
                      '429 Too Many Requests\nYou have sent too many requests in a given amount of time.\n\n']
        # Check if all URLs are parsed, if not, continue retrying
        all_urls_parsed = True if len(error_urls) == 0 else False
        sitemap_loader.filter_urls = error_urls
    return final_docs

# Function to refresh vector embeddings for the crawled documents


@st.cache_resource(ttl=86400, show_spinner="Refreshing the data embeddings. This may take a while...")
def refresh_embeddings(main_url):
    # Initialize Google Cloud AI Platform with project ID and location
    aiplatform.init(project=f"{project_id}", location=f"us-central1")
    # Get all the crawled documents
    documents = get_docs(main_url)
    print(f"Total docs = {len(documents)}")
    # Initialize VertexAIEmbeddings
    embeddings = VertexAIEmbeddings()
    # Split documents into chunks for indexing
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator=" ",)
    chunked_docs = text_splitter.split_documents(documents)
    try:
        # Create and save FAISS index for chunked documents
        faiss_index = FAISS.from_documents(chunked_docs, embeddings)
        faiss_index.save_local("index/" + get_base_url(main_url))
    except:
        # Display error if the URL is not valid or indexing fails
        st.error(
            "Please verify the URL provided. It must be a link to the website's Sitemap")

# Function to fetch possible responses based on user's query and similarity threshold


def fetch_result_set(query, similarity_threshold, main_url):
    # Initialize Google Cloud AI Platform with project ID and location
    aiplatform.init(project=f"{project_id}", location=f"us-central1")
    # Initialize VertexAIEmbeddings for similarity search
    embeddings = VertexAIEmbeddings()
    # Load the FAISS index for the main URL
    vdb_chunks = FAISS.load_local(
        "index/" + get_base_url(main_url), embeddings)
    # Perform similarity search with user's query
    results = vdb_chunks.similarity_search_with_score(query)
    matches = []
    # Check if any results are found
    if len(results) == 0:
        raise Exception(
            "Did not find any results. Adjust the query parameters.")
    # Append matched documents to the matches list with similarity score
    for r in results:
        matches.append(
            {
                "source": r[0].metadata["source"],
                "content": r[0].page_content,
                "similarity": round(r[1], 2),
            }
        )
    matches = pd.DataFrame(matches)
    return matches

# Function to run the summarization and document retrieval chain with user's query and matched documents


def run_chain(query, matches):
    # Initialize VertexAI for the summarization chain
    llm = VertexAI()
    # Define prompt templates for summarization and document retrieval
    map_prompt_template = """
                  You will be asked a question.
                  This question is enclosed in triple backticks (```).
                  Generate a summary with all details along with the sources of information.
                  ```{text}```
                  SUMMARY:
                  """
    map_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                    You will be given a question and set of possible answers
                    enclosed in triple backticks (```) .
                    You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way.
                    Select the response that is most relevant to answer the question.
                    question in as much detail as possible.
                    Your answer should include the exact answer and related details. 
                    Your answer should be in Markdown in a numbered list format.


                    Description:
                    ```{text}```


                    Question:
                    ``{query}``


                    Answer:
                    """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text", "query"]
    )
    docs = [Document(page_content=t) for t in matches["content"]]

    # Load the summarization and document retrieval chain
    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt

    )

    # Generate response by running the chain with the user's query and matched documents
    answer = chain.run(
        {
            "input_documents": docs,
            "query": query,
        }
    )

    return answer


def main():
    # Streamlit app title and expander for defining or updating the knowledge base
    st.title("A GenAI Based Custom Search Engine")
    my_expander = st.expander(
        label='Click here to define or update the knowledge base', expanded=False)
    with my_expander:
        input_url = st.text_input(
            "Enter the Sitemap URL of the knowledge base")
        similarity_threshold = st.number_input(
            "Enter the similarity threshold", value=0.5)
        regenerate = st.checkbox(
            "Do you want to regenerate the vectors for the data?", value=False)
        st.write(
            "**If the index for the website doesn't exist, it will get generated during the first run")

    # User input for the query
    query = st.text_input("Ask a question:", "What is this all about?")

    if st.button("Get Answer"):
        # Validate and standardize the URL, and check if the index needs to be regenerated
        msg, exit_code, url = validate_url_scheme(input_url)
        if exit_code == 1:
            st.error(msg)
        else:
            if regenerate:
                # Refresh the vector embeddings if requested
                refresh_embeddings(url)
                st.success("Vector Embedding Complete!")
            # Fetch possible responses based on the user's query and similarity threshold
            with st.spinner(f"""Fetching possible responses with similarity={similarity_threshold}"""):
                results = fetch_result_set(query, similarity_threshold, url)
                final = run_chain(query, results)
                st.success("Completed")
            # Display the generated responses
            st.markdown(final)


# Run the Streamlit app
if __name__ == "__main__":
    main()
