

"""
Deploy the RFID Knowledge Assistant (own RAG — Vertex AI RAG Engine corpus)
to Vertex AI Agent Engine.

No Dockerfile, no Cloud Build, no Cloud Run, no Streamlit - Agent Engine
builds and hosts this for you. Once deployed, register it in Gemini
Enterprise to get the chat UI, auth, and logging for free.
"""


# Make sure the agent module sees it too, since rfid_agent.py reads from env

import vertexai
from vertexai import agent_engines

import os


RAG_CORPUS_NAME = os.environ.get("RAG_CORPUS_NAME", "")

PROJECT_ID = "rta-genai-explorations-406d"
REGION = "europe-west1"
STAGING_BUCKET = "gs://rta-genai-explorations-406d-staging"
RAG_CORPUS_NAME = "projects/856095474659/locations/europe-west1/ragCorpora/2227030015734710272"

os.environ["RAG_CORPUS_NAME"] = RAG_CORPUS_NAME
os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["REGION"] = REGION

from rfid_agent import root_agent  # the module above


PROJECT_ID = "rta-genai-explorations-406d"
REGION = "europe-west1"
# Must exist before deploying: gsutil mb -l europe-west1 gs://<bucket-name>
STAGING_BUCKET = "gs://rta-genai-explorations-406d-staging"

# Resource name of the corpus you created with rag.create_corpus() +
# rag.import_files() in this same project, e.g.:
# "projects/<project-number>/locations/europe-west1/ragCorpora/<id>"
#RAG_CORPUS_NAME = "projects/856095474659/locations/europe-west1/ragCorpora/4611686018427387904"
#Corpus resource name: projects/856095474659/locations/europe-west1/ragCorpora/4611686018427387904

# 1. Wrap the ADK agent for Agent Engine
app = agent_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

# 2. Local smoke test before deploying
print("--- Local smoke test ---")
for event in app.stream_query(
    user_id="local-test-user",
    message="What is RFID and how is it used in this process?",
):
    print(event)

# 3. Deploy to Agent Engine
vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

remote_agent = agent_engines.create(
    agent_engine=app,
   requirements=[
        "google-cloud-aiplatform[agent_engines,adk,rag]==1.162.0",
        "google-genai==2.11.0",
        "google-adk==2.4.0",
        "pydantic==2.13.4",
        "cloudpickle==3.1.2",
    ],
    extra_packages=["rfid_agent.py"],  # bundle the local module so it can be unpickled remotely
    display_name="RFID Knowledge Assistant (Own RAG)",
    env_vars={
        "PROJECT_ID": PROJECT_ID,
        "REGION": REGION,
        "RAG_CORPUS_NAME": RAG_CORPUS_NAME,
        "RAG_TOP_K": "3",
    },
)

print("Deployed. Resource name:")
print(remote_agent.resource_name)
# -> projects/<PROJECT_NUMBER>/locations/europe-west1/reasoningEngines/<ID>
# Use this to register the agent in Gemini Enterprise.

# 4. Remote smoke test
print("\n--- Remote smoke test ---")
for event in remote_agent.stream_query(
    user_id="remote-test-user",
    message="Summarize the key RFID process steps from the presentation.",
):
    print(event)