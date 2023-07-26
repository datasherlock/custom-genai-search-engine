## Introduction
This tool creates a custom search engine using VertexAI, Langchain and Streamlit. It allows users to input the URL of a website's sitemap XML file, which will serve as the knowledge base. The app then crawls the entire website, refreshes vector embeddings, and uses the information as a knowledge base to answer user queries.

## Architecture
![image](https://github.com/datasherlock/custom-genai-search-engine/assets/11506905/116084e2-e4d4-4a5f-a014-8920c4077dec)



## Dependencies
Before we dive into the code, it's essential to understand the dependencies used in the application:
- `beautifulsoup4`: A library for parsing HTML and XML documents.
- `langchain`: A custom library that provides various functionalities for working with natural language data, embeddings, and AI models.
- `google-cloud-aiplatform`: The official Python library for Google Cloud AI Platform, which allows us to interact with the Vertex AI service.
- `streamlit`: The framework used for creating the web application.

## Functions and Logic

### 1. `validate_url_scheme(url)`

This function takes a URL as input and checks if it starts with 'http' or 'https.' If it doesn't, it returns an error message. If the URL starts with the correct scheme, it standardizes the URL and returns it. This function ensures that the user-provided URL is valid and starts with the correct scheme.

### 2. `standardise_url(url)`

This function takes a URL as input and standardizes it by adding 'http://' or 'https://' if it's missing. It returns the standardized URL.

### 3. `get_base_url(url)`

This function takes a URL as input and extracts the base URL (excluding the protocol) from it. For example, if the input is 'https://www.example.com/page', the function will return 'www.example.com'.

### 4. `get_docs(url)`

This function takes the main website URL (sitemap XML URL) as input. It uses the `langchain` library's `SitemapLoader` to crawl the website and fetch all the documents. If rate-limiting (429 error) occurs during crawling, it retries up to a specified number of times with a backoff delay. It returns a list of document objects representing the website pages.

### 5. `refresh_embeddings(main_url)`

This function is responsible for refreshing the vector embeddings for the knowledge base. It calls the `get_docs` function to fetch all the documents from the website, then it uses the `VertexAIEmbeddings` class from the `langchain` library to create embeddings for the documents. Finally, it saves the embeddings using the `FAISS` (Facebook AI Similarity Search) index.

### 6. `fetch_result_set(query, similarity_threshold, main_url)`

This function takes the user's query, similarity threshold, and main website URL as inputs. It initializes the Vertex AI service using the `google-cloud-aiplatform` library. It loads the embeddings index for the website using `FAISS`. Then, it performs a similarity search with the user's query to find matching documents from the knowledge base based on the specified similarity threshold. The function returns a DataFrame containing the matching documents and their similarity scores.

### 7. `run_chain(query, matches)`

This function takes the user's query and the DataFrame containing matching documents and their similarity scores. It sets up a language model chain using the `langchain` library to generate a summary of the relevant documents that best answer the user's query. The function returns the generated summary in Markdown format.

## Deployment

### - Generate an image and store in GCP's Container Registry 
`gcloud builds submit --tag gcr.io/<project>/<image-name>`
Refer to the DockerFile for any specific settings you may want to modify

### - Deploy the image to a Cloud Run service in port 8501
`gcloud run deploy <service name> --image gcr.io/<project>/<image-name> --region <region> --platform managed --allow-unauthenticated --quiet --port 8501`

### - Create an HTTPS Load Balancer

### - Secure with IAP

## References

For further information on the dependencies and libraries used in the application, refer to the official documentation and resources provided below:
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Requests: https://docs.python-requests.org/en/latest/
- Streamlit: https://docs.streamlit.io/
- Google Cloud AI Platform: https://cloud.google.com/ai-platform
- Langchain: (No public documentation available. If you have access, refer to the library's official documentation)
