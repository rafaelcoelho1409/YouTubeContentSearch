from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import streamlit as st
import requests
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
from langgraph.checkpoint.memory import MemorySaver
from models.youtube_content_search import YouTubeContentSearch

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
    return agents_config

@app.post("/youtube_content_search")
def load_model(settings: dict):
    for key, value in agents_config.api_key["api_key"].items():
        os.environ[key] = value
    global agent, shared_memory
    shared_memory = MemorySaver()
    agent = YouTubeContentSearch(
        agents_config.framework,
        agents_config.temperature_filter,
        agents_config.model_name,
        shared_memory
        )
    agent.load_model(**settings)
    return "Model loaded with success."

class SearchRequest(BaseModel):
    context_to_search: str

@app.post("/youtube_content_search/context_to_search")
def get_context_to_search(request: SearchRequest):
    events = agent.stream_graph_updates(request.context_to_search)
    return events

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

class YoutubeSearchAgentRequest(BaseModel):
    user_input: str

@app.post("/youtube_content_search/youtube_search_agent")
def build_youtube_search_agent(question: YoutubeSearchAgentRequest):
    response = agent.llm.invoke({"user_input": question.user_input})
    return {"answer": response.content}
    #for key, value in agents_config.api_key["api_key"].items():
    #    os.environ[key] = value
    #try:
    #    prompt = ChatPromptTemplate.from_messages(
    #        [
    #            (
    #                "system",
    #                """
    #                You are a YouTube search agent.\n
    #                Based on the following prompt:\n\n
    #                {user_input}\n\n
    #                You must take this user input and transform it into 
    #                the most efficient youtube search query you can.\n
    #                It must be in a string format.\n
    #                """,
    #            ),
    #            ("placeholder", "{messages}"),
    #        ]
    #    )
    #    chain = prompt | agent.llm.with_structured_output(SearchQuery)
    #    response = chain.invoke({"user_input": request.user_input})
    #    return response.dict()
    #except Exception as e:
    #    raise HTTPException(
    #        status_code = 500,
    #        detail = {
    #            "status": "error",
    #            "message": str(e),
    #            "type": type(e).__name__
    #        }
    #    )