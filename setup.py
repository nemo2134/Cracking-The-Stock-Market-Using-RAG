import utils
# import Tru
import os
import openai
openai.api_key = utils.get_openai_api_key()
from llama_index import SimpleDirectoryReader
#Add additional files here
documents = SimpleDirectoryReader(
    input_files=["Identifying-Chart-Patterns.pdf","The Most Important Thing Uncommon Sense for the Thoughtful Investor.pdf"]
).load_data()
#Basic RAG Pipeline
from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)
query_engine = index.as_query_engine()

test_query = "What are the steps to become a successful stock trader?"
response = query_engine.query(
    test_query
)
print("\n" + test_query+"\n" + "\n" +str(response) + "\n")

#Evaluation setup using TruLens
eval_questions = []
#Include file with your evaluation questions
# with open('eval_questions.txt', 'r') as file:
#     for line in file:
#         # Remove newline character and convert to integer
#         item = line.strip()
#         print(item)
#         eval_questions.append(item)

# Add your own question
new_question = "What are some chart patterns to know before trading stocks?"
eval_questions.append(new_question)

from trulens_eval import Tru
tru = Tru()
tru.reset_database()
from utils import get_prebuilt_trulens_recorder
tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                            app_id="Direct Query Engine")
with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)
records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()
tru.run_dashboard()

