from setup import documents, eval_questions, tru
from utils import get_prebuilt_trulens_recorder, build_automerging_index

#Auto-Merging Retrieval
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)
from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
auto_merging_response = automerging_query_engine.query(
    "How does news and market sentiment affect stock prices?"
)
test_question = "How does news and market sentiment affect stock prices?"
print("\n" + test_question+"\n" + "\n" +str(auto_merging_response) + "\n")
tru.reset_database()
tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                        app_id="Automerging Query Engine")
test_question = "What is technical analysis and fundamental analysis?"
eval_questions.append(test_question)
for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])
tru.run_dashboard()