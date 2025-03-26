from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import streamlit as st
import stqdm
import os
import uuid
import re
import pandas as pd
import requests
import json
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import TokenTextSplitter
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from neo4j import GraphDatabase
from pytubefix import YouTube, Channel, Playlist
from pytubefix.contrib.search import Search, Filter
from models.youtube_content_search import YouTubeContentSearch

#------------------------------------------------
###STRUCTURES
class SearchQuery(BaseModel):
    search_query: str = Field(
        #"An accurate search query for YouTube videos.",
        title = "Search Query",
        description = """
        Search query for YouTube videos. 
        Must be only 1 search query at the maximum.
        You must provide a search query with the best keywords that is accurate and concise.
        It can be an extense search query if necessary, but it must be only one.
        Give the best and most accurate search query you can.
        """,
        example = "How to make a cake"
    )


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description = "All the person, organization, or business entities that "
        "appear in the text",
    )
#------------------------------------------------
app = FastAPI()


@app.get("/settings/frameworks")
def get_framework():
    return {
        "Groq": "GROQ_API_KEY",
        "SambaNova": "SAMBANOVA_API_KEY",
        "Scaleway": [
            "SCW_GENERATIVE_APIs_ENDPOINT",
            "SCW_ACCESS_KEY",
            "SCW_SECRET_KEY"
        ],
        "OpenAI": "OPENAI_API_KEY"
    }

@app.get("/settings/frameworks/models")
def get_framework_models():
    return {
        "Groq": [
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b-specdec",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-specdec",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
                ], 
        "Google Generative AI": [
            "gemini-1.5-pro",
            #"gemini-2.0-flash"
        ],
        "SambaNova": [
            "DeepSeek-R1",
            "DeepSeek-R1-Distill-Llama-70B",
            "Llama-3.1-Tulu-3-405B",
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.1-70B-Instruct",
            "Meta-Llama-3.1-8B-Instruct",
            "Meta-Llama-3.3-70B-Instruct",
            "Meta-Llama-Guard-3-8B",
            "Qwen2.5-72B-Instruct",
            "Qwen2.5-Coder-32B-Instruct",
            "QwQ-32B-Preview"
        ],
        "Scaleway": [
            "deepseek-r1",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-instruct",
            "llama-3.1-70b-instruct",
            "llama-3.1-8b-instruct",
            "mistral-nemo-instruct-2407",
            "pixtral-12b-2409",
            "qwen2.5-coder-32b-instruct",
            "bge-multilingual-gemma2"
        ],
        "OpenAI": [
            "gpt-4o",
            "chatgpt-4o-latest",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
            "o1-preview"
        ]
    }

@app.get("/search_type")
def get_search_type():
    return [
    "Search",
    "Video", 
    "Channel", 
    "Playlist"
    ]

@app.get("/search/upload_date")
def get_search_upload_date():
    return [
        None,
        "Last Hour",
        "Today",
        "This Week",
        "This Month",
        "This Year"
        ]

@app.get("/search/duration")
def get_search_duration():
    return [
        None,
        "Under 4 minutes",
        "Over 20 minutes",
        "4 - 20 minutes"
        ]

@app.get("/search/features")
def get_search_duration():
    return [
        "Live",
        "4K",
        "HD",
        "Subtitles/CC",
        "Creative Commons",
        "360",
        "VR180",
        "3D",
        "HDR",
        "Location",
        "Purchased"
    ]

@app.get("/search/sort_by")
def get_search_duration():
    return [
        None,
        "Relevance",
        "Upload Date",
        "View count",
        "Rating"
    ]

class AgentsConfig(BaseModel):
    framework: str | None
    temperature_filter: float | None
    model_name: str | None
    api_key: dict | None

agents_config = AgentsConfig(
    framework = None,
    temperature_filter = None,
    model_name = None,
    api_key = {"api_key": None})

@app.get("/agents_config", response_model = AgentsConfig, response_model_exclude = {"api_key"})
def get_agents_config():
    return agents_config

@app.put("/agents_config", response_model = AgentsConfig, response_model_exclude = {"api_key"})
def update_agents_config(config: AgentsConfig):
    agents_config.framework = config.framework
    agents_config.temperature_filter = config.temperature_filter
    agents_config.model_name = config.model_name
    agents_config.api_key = config.api_key
    for key, value in agents_config.api_key["api_key"].items():
        os.environ[key] = value
    return agents_config

@app.get("/youtube_content_search/clear_neo4j_graph")
def clear_neo4j_graph():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth = (
            os.getenv("NEO4J_USERNAME"), 
            os.getenv("NEO4J_PASSWORD")
            )
        )
    with driver.session(database = "neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
    return "All previous Neo4J relationships cleared to avoid context confusion."

@app.get("/youtube_content_search")
def set_model():
    global shared_memory, config, llm
    shared_memory = MemorySaver()
    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [StreamlitCallbackHandler(st.container())]}
    llm_framework = {
        "Groq": ChatGroq,
        "Ollama": ChatOllama,
        "Google Generative AI": ChatGoogleGenerativeAI,
        "SambaNova": ChatSambaNovaCloud,
        "Scaleway": ChatOpenAI,
        "OpenAI": ChatOpenAI,
    }
    llm_model = llm_framework[agents_config.framework]
    if agents_config.framework == "Scaleway":
        llm = ChatOpenAI(
            base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
            api_key = os.getenv("SCW_SECRET_KEY"),
            model = agents_config.model_name,
            temperature =  agents_config.temperature_filter
        )
    else:
        try:
            llm = llm_model(
                model = agents_config.model_name,
                temperature = agents_config.temperature_filter,
            )
        except:
            llm = llm_model(
                model = agents_config.model_name,
                #temperature = agents_config.temperature_filter,
            )
    return "Model set with success."

@app.get("/youtube_content_search/load_model")
def load_model():
    requests.get("http://fastapi:8000/youtube_content_search/clear_neo4j_graph")
    global neo4j_graph, vector_index, llm_transformer
    neo4j_graph = Neo4jGraph()
    vector_index = Neo4jVector.from_existing_graph(
        HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"),
        search_type = "hybrid",
        node_label = "Document",
        text_node_properties = ["text"],
        embedding_node_property = "embedding"
    )
    llm_transformer = LLMGraphTransformer(llm = llm)
    return "Model loaded with success."

class YouTubeSearchAgentRequest(BaseModel):
    user_input: str

@app.post("/youtube_content_search/youtube_search_agent")
def build_youtube_search_agent(request: YouTubeSearchAgentRequest):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a YouTube search agent.\n
                Based on the following prompt:\n\n
                {user_input}\n\n
                You must take this user input and transform it into 
                the most efficient youtube search query you can.\n
                It must be in a string format.\n
                """,
            ),
            ("placeholder", "{messages}"),
        ]
    )
    chain = prompt | llm.with_structured_output(SearchQuery)
    response = chain.invoke({"user_input": request.user_input})
    return {"search_query": response.search_query}

class EntityChainRequest(BaseModel):
    question: str

@app.post("/youtube_content_search/entity_chain")
def build_entity_chain(request: EntityChainRequest):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    global entity_chain
    entity_chain = prompt | llm.with_structured_output(Entities)
    entities = entity_chain.invoke({"question": request.question})
    return {"names": entities.names}

class RAGChainRequest(BaseModel):
    user_input: str

@app.post("/youtube_content_search/rag_chain")
def build_rag_chain(request: RAGChainRequest):
    ###TOOLS
    def _format_chat_history(chat_history):
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content = human))
            buffer.append(AIMessage(content = ai))
        return buffer
    #-------------------------------------------
    def generate_full_text_query(input):
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    #-------------------------------------------
    # Fulltext index query
    def structured_retriever(question):
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        #entities = entity_chain.invoke({"question": question})
        entities = requests.post(
            "http://fastapi:8000/youtube_content_search/entity_chain",
            json = {"question": question}
        ).json()
        for entity in entities["names"]:
            response = neo4j_graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    #-------------------------------------------
    # Retriever
    def retriever(question):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data
    #-------------------------------------------
    # Condense a chat history and follow-up question into a standalone question
    _template = """
        Given the following conversation and a follow up question, rephrase 
        the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name = "HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history = lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm#ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    question_answer = rag_chain.invoke({"question": request.user_input})
    return {"question_answer": question_answer}
