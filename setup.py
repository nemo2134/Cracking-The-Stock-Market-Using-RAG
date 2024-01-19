import utils
# import Tru
import os
import openai
import pandas as pd
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
response = query_engine.query(
    "What are the steps to become a successful stock trader?"
)
print("\n"+str(response))

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
new_question = "What is the right brokerage for me?"
eval_questions.append(new_question)
print(eval_questions)
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
# records.head()
# tru.run_dashboard()


#Displaying the record results using DataFrame and Dash Dashboard
df_records = pd.DataFrame(records)
# Comment out if you want the record_json column
if 'record_json' in df_records.columns:
    df_records = df_records.drop(columns=['record_json'])
# Comment out if you want the app_json column
if 'app_json' in df_records.columns:
    df_records = df_records.drop(columns=['app_json'])

import dash
from dash import dash_table
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Records Dashboard"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df_records.columns],
        data=df_records.to_dict('records'),
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)


from selenium import webdriver
import time

def capture_dashboard():
    # Set the path for your WebDriver
    driver = webdriver.Chrome('path/to/chromedriver')
    driver.get('http://localhost:8050')  # The URL where your dash app is running
    time.sleep(5)  # Wait for the page to load
    driver.save_screenshot('dashboard.png')
    driver.quit()

import tkinter as tk
from PIL import Image, ImageTk

def show_popup():
    root = tk.Tk()
    root.title("Dashboard Record")
    image = Image.open("dashboard.png")
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()
    root.mainloop()

capture_dashboard()
show_popup()
