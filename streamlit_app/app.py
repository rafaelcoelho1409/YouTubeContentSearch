import streamlit as st
import requests
from langgraph.checkpoint.memory import MemorySaver
from functions import (
    settings,
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graphs,
    view_neo4j_context_graph,
)
from models.youtube_content_search import (
    YouTubeContentSearch, 
    YouTubeChatbot
)


st.set_page_config(
    page_title = "YouTube Content Search",
    layout = "wide",
    initial_sidebar_state = "expanded")

st.sidebar.title("YouTube Content Search")
st.sidebar.caption("Author: Rafael Silva Coelho")

settings_button = st.sidebar.button(
    label = "Settings",
    use_container_width = True
)
if settings_button:
    settings()
st.session_state["view_graph_button_container"] = st.sidebar.container()
        

initialize_shared_memory()


clear_memory_button = st.sidebar.button(
    label = "Clear memory",
    use_container_width = True,
)
if clear_memory_button:
    if "youtube_agent" in st.session_state:
        st.session_state["youtube_agent"].shared_memory = MemorySaver()
    st.session_state["snapshot"] = []
    requests.put(
        "http://fastapi:8000/streamlit_actions",
        json = {"streamlit_actions": []}
    )


if not "snapshot" in st.session_state:
    st.session_state["snapshot"] = []
streamlit_actions = requests.get(
    "http://fastapi:8000/streamlit_actions"
).json()
st.session_state["snapshot"] += streamlit_actions

for actions in st.session_state["snapshot"]:
    if actions != []:
        for action in actions:
            st.chat_message(
                action[3]
            ).expander(
                action[2][0], 
                expanded = action[2][1]
            ).__getattribute__(
                action[0]
            )(
                **action[1]
            )

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


search_type_filter = st.sidebar.selectbox(
    label = "Search type",
    options = requests.get("http://fastapi:8000/search_type").json(),
    index = 0
)

if search_type_filter == "Search":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on YouTube",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            max_value = 20,
            step = 1,
            value = 1
        )
        upload_date = st.selectbox(
            label = "Upload date",
            options = requests.get("http://fastapi:8000/search/upload_date").json(),
            index = 0
        )
        duration = st.selectbox(
            label = "Duration",
            options = requests.get("http://fastapi:8000/search/duration").json(),
            index = 0
        )
        features = st.multiselect(
            label = "Features",
            options = requests.get("http://fastapi:8000/search/features").json(),
            default = []
        )
        sort_by = st.selectbox(
            label = "Sort by",
            options = requests.get("http://fastapi:8000/search/sort_by").json(),
            index = 2
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Video":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on YouTube",
        )
        video_url = st.text_input(
            label = "Video URL",
            placeholder = "Provide the URL of the video to search",
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Channel":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on this YouTube Channel",
        )
        channel_url = st.text_input(
            label = "Channel URL",
            placeholder = "Provide the URL of the channel to search",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            step = 1,
            value = 5
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Playlist":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on this YouTube playlist",
        )
        playlist_url = st.text_input(
            label = "Playlist URL",
            placeholder = "Provide the URL of the playlist to search",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            step = 1,
            value = 5
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)


if submit_project_settings:
    settings_dict = {
        "max_results": None,
        "search_type": None,
        "upload_date": None,
        "video_type": None,
        "duration": None,
        "features": None,
        "sort_by": None,
        "video_url": None,
        "channel_url": None,
        "playlist_url": None,
    }
    st.session_state["youtube_agent"] = YouTubeContentSearch()
    if search_type_filter == "Search":
        settings_dict["max_results"] = max_results
        settings_dict["search_type"] = search_type_filter
        settings_dict["upload_date"] = upload_date
        settings_dict["duration"] = duration
        settings_dict["features"] = features
        settings_dict["sort_by"] = sort_by
    elif search_type_filter == "Video":
        settings_dict["video_url"] = video_url
        settings_dict["search_type"] = search_type_filter
    elif search_type_filter == "Channel":
        settings_dict["channel_url"] = channel_url
        settings_dict["search_type"] = search_type_filter
        settings_dict["max_results"] = max_results
    elif search_type_filter == "Playlist":
        settings_dict["playlist_url"] = playlist_url
        settings_dict["search_type"] = search_type_filter
        settings_dict["max_results"] = max_results
    #Updating model_config
    requests.put(
        "http://fastapi:8000/model_config",
        json = settings_dict
    )
    #Loading YouTubeContentSearch
    with st.spinner("Loading YouTube Content Search..."):
        requests.get(
            "http://fastapi:8000/youtube_content_search",
        )
    with st.spinner("Loading model..."):
        requests.get(
            "http://fastapi:8000/youtube_content_search/load_model",
        )
    st.session_state["youtube_agent"].build_graph()
    st.session_state["youtube_agent"].stream_graph_updates(context_to_search)
    st.session_state["context_to_search"] = context_to_search

try:
    context_to_search = st.session_state["context_to_search"]
except:
    st.info("Please, provide a context for YouTube searches.")
    st.stop()


st.session_state["chatbot_agent"] = YouTubeChatbot(
    st.session_state["youtube_agent"].shared_memory
)
st.session_state["chatbot_agent"].load_model()


view_app_graph = st.session_state["view_graph_button_container"].button(
    label = "View application graphs",
    use_container_width = True,
)
if view_app_graph:
    view_application_graphs(
        {
            "YouTube Content Search": st.session_state["youtube_agent"].graph,
            "YouTube Chatbot": st.session_state["chatbot_agent"].graph})
    

view_neo4j_graph = st.sidebar.button(
    label = "View Neo4j context graph",
    use_container_width = True,
)
if view_neo4j_graph:
    view_neo4j_context_graph()


if prompt := st.chat_input():
    st.session_state["chatbot_agent"].stream_graph_updates(
        prompt)