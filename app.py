import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv, find_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
load_dotenv(find_dotenv())

## Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.title("LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Get the groq api key and url(YT or website) to be summarized
with st.sidebar:
    api_key = st.text_input("Groq API Key", value ="", type="password")
    if api_key == "Sourav":
        groq_api_key = os.getenv("GROQ_API_KEY")
    else:
        groq_api_key = api_key
generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

map_prompt = PromptTemplate(
    template="""
Summarize the following section of a YouTube transcript.
Extract the important facts, ideas, arguments, examples, and conclusions.
Do not omit important information.

Transcript section:
{text}
""",
    input_variables=["text"]
)

combine_prompt = PromptTemplate(
    template="""
Using the summaries of the transcript sections below, create one
coherent final summary of approximately 300 words.

Remove repetition and focus on the most important information.
Preserve important facts, conclusions, and key points.

Section summaries:
{text}
""",
    input_variables=["text"]
)


if st.button("Summarize the content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the informaton to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting.....'"):
                # Loading the website or YT video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    # Extract YouTube video ID
                    if "youtu.be" in generic_url:
                        video_id = urlparse(generic_url).path.lstrip("/")
                    else:
                        video_id = parse_qs(urlparse(generic_url).query)["v"][0]

                    st.info("Fetching YouTube transcript...")

                    # Fetch Hindi or English transcript
                    api = YouTubeTranscriptApi()

                    transcript = api.fetch(
                        video_id,
                        languages=["hi", "en"]
                    )

                    # Convert transcript snippets into plain text
                    transcript_text = " ".join(
                        snippet.text for snippet in transcript
                    )

                    # Convert to LangChain Document
                    data = [
                        Document(
                            page_content=transcript_text,
                            metadata={
                                "source": generic_url,
                                "video_id": video_id
                            }
                        )
                    ]
                else:
                    loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify=False,
                                                headers={ 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
                
                    data = loader.load()

                ## Chain For Summarization
                # Split transcript into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=200
                )

                docs = text_splitter.split_documents(data)

                st.info(f"Transcript split into {len(docs)} chunks")

                # Summarize chunks and combine summaries
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                output = chain.invoke(docs)

                output_summary = output["output_text"]

                st.success(output_summary)
            
        except Exception as e:
            st.exception(f"Exception:{e}")



                
                
                
