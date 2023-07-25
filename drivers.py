from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import time
import pandas  as pd
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain import FAISS
from typing import Any, Callable, Generator, Iterable, List, Optional
from langchain.document_loaders.sitemap import SitemapLoader
from urllib.parse import urlparse

NUM_RETRIES = 10
BACKOFF_FACTOR = 1

def validate_url_scheme(url) -> str:
    if not url.startswith('http'):
        msg = "The URL must begin with http:// or https://"
        exit_code = 1
        url = ""
    else:
        msg = "URL Scheme Validated"
        exit_code = 0
        url = standardise_url(url)
    return (msg, exit_code, url)

def standardise_url(url):
    parsed_url = urlparse(url)
    final_url = 'www.'+ parsed_url.netloc if not parsed_url.netloc.startswith('www.') else parsed_url.netloc
    return parsed_url.scheme + '://' + final_url + parsed_url.path

def get_base_url(url):
    parsed_url = urlparse(url)
    final_url = 'www.'+ parsed_url.netloc if not parsed_url.netloc.startswith('www.') else parsed_url.netloc
    return final_url

def get_docs(url):
    all_urls_parsed = False
    final_docs = []
    error_urls = []
    sitemap_loader = SitemapLoader(web_path=url)
    while not all_urls_parsed:
        sitemap_loader.requests_per_second = 10
        sitemap_loader.raise_for_status = True
        for retry_num in range(NUM_RETRIES):
            time_to_sleep = BACKOFF_FACTOR * (2 ** (retry_num - 1))
            try:  # try to get the response from the url
                initial_docs = sitemap_loader.load()
                if len(initial_docs) != len(error_urls) and len(error_urls) > 1 :
                    raise requests.exceptions.HTTPError
                break
            except requests.exceptions.HTTPError:  # if there is an error
                time.sleep(time_to_sleep)  # sleep
                pass

        
        final_docs.extend([ doc for doc in initial_docs if doc.page_content != '429 Too Many Requests\nYou have sent too many requests in a given amount of time.\n\n'])
        error_urls = [ doc.metadata['source'] for doc in initial_docs if doc.page_content == '429 Too Many Requests\nYou have sent too many requests in a given amount of time.\n\n']
        all_urls_parsed = True if len(error_urls) == 0 else False
        sitemap_loader.filter_urls=error_urls
    return final_docs


def retry_with_backoff(func, *args, retry_delay=5, backoff_factor=2, **kwargs):
    max_attempts = 10
    retries = 0
    for i in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"error: {e}")
            retries += 1
            wait = retry_delay * (backoff_factor**retries)
            print(f"Retry after waiting for {wait} seconds...")
            time.sleep(wait)

def refresh_embeddings(main_url):
    aiplatform.init(project=f"{project_id}", location=f"us-central1")
    documents = get_docs(main_url)
    print(f"Total docs = {len(documents)}")
    embeddings = VertexAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0,separator = " ",)
    chunked_docs = text_splitter.split_documents(documents)
    try:
        faiss_index = FAISS.from_documents(chunked_docs, embeddings)
    except:
        st.error("Please verify the URL provided. It must be a link to the website's Sitemap")
    faiss_index.save_local("index/" + get_base_url(main_url))



def fetch_result_set(query,similarity_threshold, main_url):
    aiplatform.init(project=f"{project_id}", location=f"us-central1")
    embeddings = VertexAIEmbeddings()
    vdb_chunks  =   FAISS.load_local("index/" + get_base_url(main_url), embeddings)
    results = vdb_chunks.similarity_search_with_score(query)
    matches = []
#    print(results)

    if len(results) == 0:
        raise Exception("Did not find any results. Adjust the query parameters.")

    for r in results:
        # Collect the description for all the matched similar toy products.
        matches.append(
            {
                "source": r[0].metadata["source"],
                "content": r[0].page_content,
                "similarity": round(r[1], 2),
            }
        )

    matches = pd.DataFrame(matches)

    return matches

def run_chain(query,matches):
    llm = VertexAI()

    map_prompt_template = """
                  You will be asked a question.
                  This question is enclosed in triple backticks (```).
                  Generate a summary with all details along with the sources of information.
                  ```{text}```
                  SUMMARY:
                  """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

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

    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt

    )

    #print(docs)
    answer = chain.run(
        {
            "input_documents": docs,
            "query": query,
        }
    )

    #print(matches)
    return answer

def main():
    st.title("A GenAI Based Custom Search Engine")
    my_expander = st.expander(label='Click here to define or update the knowledge base', expanded=False)
    with my_expander:
        input_url = st.text_input("Enter the Sitemap URL of the knowledge base")
        similarity_threshold=st.number_input("Enter the similarity threshold", value=0.5)
        regenerate=st.checkbox("Do you want to regenerate the vectors for the data?", value=False)
        st.write("**If the index for the website doesn't exist, it will get generated during the first run")
    
    query = st.text_input("Ask a question:","Define the entire knowledge base?")

    if st.button("Get Answer"):
        msg, exit_code, url = validate_url_scheme(input_url)
        if exit_code == 1:
            st.error(msg)
        else:
            if regenerate:
                with st.spinner("Refreshing the data embeddings. This may take a while..."):
                    refresh_embeddings(url)
                    st.success("Vector Embedding Complete!")
            with st.spinner(f"""Fetching possible responses with similarity={similarity_threshold}"""):
                results = fetch_result_set(query,similarity_threshold, url)
                final = run_chain(query,results)
                st.success("Completed")
            st.markdown(final)