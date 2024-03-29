import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# questions_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#     You are a helpful assistant that is role playing as a teacher.
         
#     Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
#     Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
#     Use (o) to signal the correct answer.
         
#     Question examples:
         
#     Question: What is the color of the ocean?
#     Answers: Red|Yellow|Green|Blue(o)
         
#     Question: What is the capital or Georgia?
#     Answers: Baku|Tbilisi(o)|Manila|Beirut
         
#     Question: When was Avatar released?
#     Answers: 2007|2001|2009(o)|1998
         
#     Question: Who was Julius Caesar?
#     Answers: A Roman Emperor(o)|Painter|Actor|Model
         
#     Your turn!
         
#     Context: {context}
#     """,
#         )
#     ]
# )
# questions_chain = {"context": format_docs} | questions_prompt | llm

# formatting_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#     You are a powerful formatting algorithm.
     
#     You format exam questions into JSON format.
#     Answers with (o) are the correct ones.
     
#     Example Input:
#     Question: What is the color of the ocean?
#     Answers: Red|Yellow|Green|Blue(o)
         
#     Question: What is the capital or Georgia?
#     Answers: Baku|Tbilisi(o)|Manila|Beirut
         
#     Question: When was Avatar released?
#     Answers: 2007|2001|2009(o)|1998
         
#     Question: Who was Julius Caesar?
#     Answers: A Roman Emperor(o)|Painter|Actor|Model
    
#     Example Output:
     
#     ```json
#     {{ "questions": [
#             {{
#                 "question": "What is the color of the ocean?",
#                 "answers": [
#                         {{
#                             "answer": "Red",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Yellow",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Green",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Blue",
#                             "correct": true
#                         }},
#                 ]
#             }},
#                         {{
#                 "question": "What is the capital or Georgia?",
#                 "answers": [
#                         {{
#                             "answer": "Baku",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Tbilisi",
#                             "correct": true
#                         }},
#                         {{
#                             "answer": "Manila",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Beirut",
#                             "correct": false
#                         }},
#                 ]
#             }},
#                         {{
#                 "question": "When was Avatar released?",
#                 "answers": [
#                         {{
#                             "answer": "2007",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "2001",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "2009",
#                             "correct": true
#                         }},
#                         {{
#                             "answer": "1998",
#                             "correct": false
#                         }},
#                 ]
#             }},
#             {{
#                 "question": "Who was Julius Caesar?",
#                 "answers": [
#                         {{
#                             "answer": "A Roman Emperor",
#                             "correct": true
#                         }},
#                         {{
#                             "answer": "Painter",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Actor",
#                             "correct": false
#                         }},
#                         {{
#                             "answer": "Model",
#                             "correct": false
#                         }},
#                 ]
#             }}
#         ]
#      }}
#     ```
#     Your turn!
#     Questions: {context}
# """,
#         )
#     ]
# )
# formatting_chain = formatting_prompt | llm

# class JsonOutputParser(BaseOutputParser):
#     def parse(self,text):
#         text = text.replace("```","").replace("json","")
#         return json.loads(text)
# output_parser = JsonOutputParser()

# @st.cache_data(show_spinner="Making Quizs...")
# def run_quiz_chain(_docs,topic):
#     # _를 붙이는거는 signature를 만들지 말라는 신호, 그래서 바뀌어도 재실행이 안됨
#     # 그럴때 매개변수를 추가하면 바뀌는 옵션을 추가할 수 있음 
#     chain = {"context":questions_chain} | formatting_chain | output_parser
#     return chain.invoke(_docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
    """,
        )
    ]
)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

output_parser = JsonOutputFunctionsParser()

@st.cache_data(show_spinner="Making Quizs...")
def run_quiz_chain(_docs,topic):
    chain = {"context": format_docs} | questions_prompt | llm.bind(function_call = {"name":"create_quiz"},functions = [function]) | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

st.set_page_config(page_title="Quiz GPT", page_icon="❓")

st.title("Quiz GPT")

with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use", ("File", "Wekipedia Article"))
    if choice == "File":
        file = st.file_uploader(
            "Upload .docx, .txt, or .pdf file", type=["pdf", "docx", "txt"]
        )
        if file:
            docs = split_file(file)

    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    st.write(response)
    with st.form("questions form"):
        for question in response["questions"]:
            value = st.radio("Select an option",[answers["answer"] for answers in question["answers"]], index = None)
            if {"answer":value,"correct": True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                correct_answer = next(answers["answer"] for answers in question["answers"] if answers["correct"])
                st.error(f"Wrong THe answer is ***{correct_answer}***")
        button = st.form_submit_button()
