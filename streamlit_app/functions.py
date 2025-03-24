import streamlit as st
import requests
import base64
import ollama
import subprocess
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
import networkx as nx
from pyvis.network import Network
from neo4j import GraphDatabase


###>>>---LOCAL FUNCTIONS---<<<###
def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: 300px;">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

def reload_active_models():
    active_models_container = st.sidebar.container()
    active_models_text = "**Active model:** "
    if st.session_state["framework"] == "Ollama":
        if ollama.ps()["models"] != []:
            for model_name in ollama.ps()["models"]:
                active_models_text += f"* {model_name['model']}\n"
        else:
            active_models_text += "No active models."
    elif st.session_state["framework"] in [
        "Groq",
        "Google Generative AI",
        "SambaNova",
        "Scaleway"
        ]:
        active_models_text += st.session_state["model_name"]
    active_models_container.info(active_models_text)

def check_model_and_temperature():
    return all([x in st.session_state.keys() for x in ["model_name", "temperature_filter"]])

def initialize_shared_memory():
    # Initialize shared memory
    if "history" not in st.session_state:
        st.session_state["history"] = StreamlitChatMessageHistory(key = "chat_history")
    if "shared_memory" not in st.session_state:
        st.session_state["shared_memory"] = MemorySaver()


###>>>---STREAMLIT FUNCTIONS---<<<###
@st.dialog("Settings", width = "large")
def settings():
    api_keys_dict = requests.get("http://fastapi:8000/settings/frameworks").json()
    framework_option = st.selectbox(
        label = "Framework",
        options = api_keys_dict.keys()
    )
    st.session_state["framework"] = framework_option
    provider_model_dict = requests.get("http://fastapi:8000/settings/frameworks/models").json()
    if framework_option == "Ollama":
        with st.form("Settings Ollama"):
            models_options = sorted(
                [x["model"] for x in ollama.list()["models"]])
            if ollama.ps()["models"] != []:
                active_models = [x["model"] for x in ollama.ps()["models"]]
                models_filter = st.selectbox(
                    label = "Ollama Models",
                    options = models_options,
                    index = models_options.index(active_models[0])
                )
            else:
                try:
                    #try to get the last model used, if exists
                    models_filter = st.selectbox(
                        label = "Ollama Models",
                        options = models_options,
                        index = models_options.index(st.session_state["model_name"])
                    )
                except:
                    models_filter = st.selectbox(
                        label = "Ollama Models",
                        options = sorted([x["model"] for x in ollama.list()["models"]])
                    )
            temperature_filter = st.slider(
                label = "Temperature",
                min_value = 0.00,
                max_value = 1.00,
                value = 0.00,
                step = 0.01
            )
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                if "model_name" in st.session_state:
                    if st.session_state["model_name"] != models_filter:
                        subprocess.run([
                            "ollama",
                            "stop",
                            st.session_state["model_name"]
                        ],
                        )
                else:
                    subprocess.run([
                        "ollama",
                        "stop",
                        models_filter
                    ],
                    )
                st.session_state["model_name"] = models_filter
                st.session_state["temperature_filter"] = temperature_filter
                try:
                    test = requests.put(
                        "http://fastapi:8000/agents_config",
                        json = {
                            "framework": st.session_state["framework"],
                            "temperature_filter": st.session_state["temperature_filter"],
                            "model_name": st.session_state["model_name"],
                            "api_key": None
                        })
                    test.raise_for_status()
                except requests.exceptions.HTTPError as err:
                    st.error(err)
                st.rerun()
    elif framework_option in [
        "Groq",
        #"Google Generative AI",
        "SambaNova",
        "Scaleway",
        "OpenAI"
    ]:
        with st.form(f"Settings {framework_option}"):
            models_option = st.selectbox(
                label = f"{framework_option} Models", 
                options = provider_model_dict[framework_option])
            temperature_filter = st.slider(
                label = "Temperature",
                min_value = 0.00,
                max_value = 1.00,
                value = 0.00,
                step = 0.01
            )
            if st.session_state["framework"] in [
                "Groq",
                #"Google Generative AI",
                "SambaNova",
                "OpenAI"
            ]:
                #AUTOFILL API KEYS, IF EXISTS
                if api_keys_dict[st.session_state["framework"]] in os.environ:
                    globals()[api_keys_dict[st.session_state["framework"]]] = st.text_input(
                        label = api_keys_dict[st.session_state["framework"]],
                        value = os.getenv(api_keys_dict[st.session_state["framework"]]),
                        placeholder = "Provide the API key",
                        type = "password"
                    )
                else:
                    globals()[api_keys_dict[st.session_state["framework"]]] = st.text_input(
                        label = api_keys_dict[st.session_state["framework"]],
                        #value = os.getenv(api_keys_dict[st.session_state["framework"]])
                        placeholder = "Provide the API key",
                        type = "password"
                    )
                api_keys_endpoint_data = {"api_key": {api_keys_dict[st.session_state["framework"]]: globals()[api_keys_dict[st.session_state["framework"]]]}}
                #os.environ[api_keys_dict[st.session_state["framework"]]] = globals()[api_keys_dict[st.session_state["framework"]]]
            elif st.session_state["framework"] == "Scaleway":
                if "SCW_GENERATIVE_APIs_ENDPOINT" in os.environ:
                    SCW_GENERATIVE_APIs_ENDPOINT = st.text_input(
                        label = "SCW_GENERATIVE_APIs_ENDPOINT",
                        value = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                        placeholder = "Provide the API endpoint",
                        type = "password"
                    )
                else:
                    SCW_GENERATIVE_APIs_ENDPOINT = st.text_input(
                        label = "SCW_GENERATIVE_APIs_ENDPOINT",
                        placeholder = "Provide the API endpoint",
                        type = "password"
                    )
                if "SCW_ACCESS_KEY" in os.environ:
                    SCW_ACCESS_KEY = st.text_input(
                        label = "SCW_ACCESS_KEY",
                        value = os.getenv("SCW_ACCESS_KEY"),
                        placeholder = "Provide the access key",
                        type = "password"
                    )
                else:
                    SCW_ACCESS_KEY = st.text_input(
                        label = "SCW_ACCESS_KEY",
                        placeholder = "Provide the access key",
                        type = "password"
                    )
                if "SCW_SECRET_KEY" in os.environ:
                    SCW_SECRET_KEY = st.text_input(
                        label = "SCW_SECRET_KEY",
                        value = os.getenv("SCW_SECRET_KEY"),
                        placeholder = "Provide the secret key",
                        type = "password"
                    )
                else:
                    SCW_SECRET_KEY = st.text_input(
                        label = "SCW_SECRET_KEY",
                        placeholder = "Provide the secret key",
                        type = "password"
                    )
                api_keys_endpoint_data = {
                    "api_key": {
                        "SCW_GENERATIVE_APIs_ENDPOINT": SCW_GENERATIVE_APIs_ENDPOINT,
                        "SCW_ACCESS_KEY": SCW_ACCESS_KEY,
                        "SCW_SECRET_KEY": SCW_SECRET_KEY}}
                #os.environ["SCW_GENERATIVE_APIs_ENDPOINT"] = SCW_GENERATIVE_APIs_ENDPOINT
                #os.environ["SCW_ACCESS_KEY"] = SCW_ACCESS_KEY
                #os.environ["SCW_SECRET_KEY"] = SCW_SECRET_KEY
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                st.session_state["model_name"] = models_option
                st.session_state["temperature_filter"] = temperature_filter
                try:
                    test = requests.put(
                        "http://fastapi:8000/agents_config",
                        json = {
                            "framework": st.session_state["framework"],
                            "temperature_filter": st.session_state["temperature_filter"],
                            "model_name": st.session_state["model_name"],
                            "api_key": api_keys_endpoint_data
                        })
                    test.raise_for_status()
                except requests.exceptions.HTTPError as err:
                    st.error(err)
                st.rerun()
            


@st.dialog("Application graph", width = "large")
def view_application_graph(graph):
    st.image(graph.get_graph(
        #xray = True
        ).draw_mermaid_png())


@st.dialog("Application graphs", width = "large")
def view_application_graphs(graph_dict):
    cols = st.columns(len(graph_dict))
    for i, (key, value) in enumerate(graph_dict.items()):
        cols[i].header(key)
        cols[i].image(value.get_graph(
            #xray = True
            ).draw_mermaid_png())


@st.dialog("Neo4J Context Graph", width = "large")
def view_neo4j_context_graph():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth = (
            os.getenv("NEO4J_USERNAME"), 
            os.getenv("NEO4J_PASSWORD")
            )
        )
    def fetch_graph_data(driver):
        """
        Retrieve nodes and relationships from Neo4j.
        Assumes that each node has an 'id' property.
        """
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        nodes, edges = {}, []
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                # Retrieve nodes from the record.
                node_a, node_b = record["n"], record["m"]
                # Use a property 'id' if available, else use Neo4j's internal id.
                id_a, id_b = node_a.get("id", node_a.id), node_b.get("id", node_b.id)
                # Store node data if not already present.
                if id_a not in nodes:
                    nodes[id_a] = dict(node_a)
                if id_b not in nodes:
                    nodes[id_b] = dict(node_b)
                # Append the edge (with relationship properties).
                edges.append((id_a, id_b, dict(record["r"])))
        return nodes, edges
    nodes_data, edges_data = fetch_graph_data(driver)
    #Build a NetworkX graph from the fetched data
    # Create a directed graph (use nx.Graph() for undirected).
    G = nx.DiGraph()
    # Add nodes along with their properties.
    for node_id, properties in nodes_data.items():
        G.add_node(node_id, **properties)
    # Add edges along with any relationship properties.
    for source, target, rel_props in edges_data:
        G.add_edge(source, target, **rel_props)
    #Create an interactive Pyvis visualization from the NetworkX graph
    # Initialize a Pyvis Network.
    pyvis_net = Network(height = "600px", width = "100%", directed = True)
    # Load the NetworkX graph into the Pyvis network.
    pyvis_net.from_nx(G)
    # (Optional) Apply a layout algorithm for better visualization.
    pyvis_net.force_atlas_2based()
    # Instead of opening a browser, save the Pyvis network as an HTML file.
    pyvis_net.save_graph("assets/graph.html")
    # Read the saved HTML file.
    with open("assets/graph.html", "r", encoding = "utf-8") as f:
        html_graph = f.read()
    st.title("Interactive Neo4j Graph Visualization")
    # Embed the Pyvis-generated HTML in your Streamlit app.
    st.components.v1.html(html_graph, height = 600, scrolling = True)


###>>>---CACHE FUNCTION---<<<###