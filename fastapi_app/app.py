from fastapi import FastAPI
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