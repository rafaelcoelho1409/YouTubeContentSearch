![YouTube Content Search](assets/YouTubeContentSearch_logo.png)
# YouTube Content Search
Demonstration: [YouTube Content Search Presentation](https://rafaelcoelho1409.github.io/uploads/YouTubeContentSearch.pdf)  
Author: Rafael Coelho ([Portfolio Website](https://rafaelcoelho1409.github.io)) 

<br><br>
<b>YouTube Content Search</b> is a platform that leverages the power of AI Agents using LangChain and LangGraph to get important informations from YouTube videos, by extracting relationships between entities (Knowledge Graph) identified from video transcriptions and by storing these informations into a Graph Database called Neo4J.<br><br>
<b>Knowledge Graphs</b> enable more precise information retrieval than traditional RAGs with vector databases by explicitly modeling relationships between entities. Unlike vector-based RAG, which relies on semantic similarity, KGs structurally capture connections (e.g., 'X works_for Y → Y located_in Z'), allowing for accurate, multi-hop reasoning and logically consistent answers. This makes KGs superior for complex queries requiring relational understanding.<br><br>
<hr>
YouTube Content Search is a platform that connects the user to open source Large Language Models like Llama 3.1 (Meta), Gemma 2 (Google), and others through services like Groq and OpenAI. By using this platform, you can search for informations through 4 options:<br>
<b>1) Search</b>: The Search option calls AI Agents to search for videos autonomously, based on the context and filters the user provides.<br>
<b>2) Video</b>: The Video option extract the informations from a specific video requested by user.<br>
<b>3) Channel</b>: The Channel option calls AI Agents to retrieve informations from video transcriptions of a YouTube channel that the user has interest to know more about.<br>
<b>4) Playlist</b>: The Playlist option works in a similar way that Channel options does, but retrieving transcription informations coming from YouTube playlists instead.<br><br>
And all these options allows the user to ask anything it wants about the videos transcriptions obtained and processed by AI Agents, and one of these AI Agents will answer accurately the questions by consulting and retrieving its own Knowledge Graph.

---

## Details about the project  

YouTube Content Search allows you to use 4 different APIs services to function with AI Agents, which you need to get an API key in order to use the project:  
- [Groq](https://console.groq.com)
- [SambaNova](https://cloud.sambanova.ai/)
- [ScaleWay](https://account.scaleway.com/)
- [OpenAI](https://platform.openai.com/)

---

## How to run this project

**A) Running through Docker Compose**  

A.1) [Install Docker](https://docs.docker.com/get-started/get-docker/)  

A.2) [Install Docker Compose](https://docs.docker.com/compose/install/)  

A.3) Clone this repository:  
> git clone https://github.com/rafaelcoelho1409/YouTubeContentSearch  

A.4) Enter this repository folder:  
> cd YouTubeContentSearch

A.5) Run Docker Compose to start the services:  
> docker compose up --build  

A.6) After all services started, you can access each of them by accessing the following local addresses:  
    - Streamlit: http://localhost:8501  
    - FastAPI: http://localhost:8000/docs  
    - MLflow: http://localhost:5000  
    - Neo4J: http://localhost:7474  
  
  
**B) Running through Kubernetes and Minikube**  
B.1) [Install Docker](https://docs.docker.com/get-started/get-docker/)  

B.2) [Install Kubernetes](https://kubernetes.io/docs/setup/)  

B.3) [Install Minikube](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download)  

B.4) Clone this repository:
> git clone https://github.com/rafaelcoelho1409/YouTubeContentSearch  

B.5) Enter this repository folder:  
> cd YouTubeContentSearch    

B.6) Start Minikube nodes:  
> minikube start  

B.7) Start Docker into Minikube:  
> eval $(minikube docker-env)  

B.8) Build all Docker images from each Dockerfile into the folders, without start them:  
> docker compose build  

B.9) Apply Kubernetes manifest to run all services on Minikube through Kubernetes:  
> kubectl apply -f k8s-manifest-minikube.yml  

B.10) You can monitor the Kubernetes Pods deployment and services running with Minikube Dashboard (open another terminal to run this command):  
> minikube dashboard  

B.11) Before accessing all the deployed services, you must get your local Minikube IP:  
> minikube ip  

B.12) With your local Minikube IP, after all services started, you can access each of them by accessing the following local addresses (replace <|minikube-ip|> for the IP you got from minikube ip command):  
    - Streamlit: http://<|minikube-ip|>:30003  
    - FastAPI: http://<|minikube-ip|>:30002/docs  
    - MLflow: http://<|minikube-ip|>:30004  
    - Neo4J: http://<|minikube-ip|>:30001  

B.13) If you prefer, you can open each service using Minikube service command:  
    - Streamlit: minikube service streamlit  
    - FastAPI: minikube service fastapi  
    - MLflow: minikube service mlflow  
    - Neo4J: minikube service neo4j  
