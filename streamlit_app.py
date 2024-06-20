import streamlit as st 
import create_database
from secret_key import hugging_face_api_key,openai_api_key
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings,OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.llms import OpenAI ,HuggingFaceHub 
from langchain.chains import LLMChain, ConversationChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import Literal
from dataclasses import dataclass
from io import StringIO
import os 

HUGGINGFACEHUB_API_TOKEN = hugging_face_api_key

CHROMA_PATH='Chroma'

REPO_ID = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
PROMPT_TEMPLATE_WELCOME = """
Read the following context:

{context}

---
Give the output based on the above context ONLY as follows. DO NOT PRINT ANYTHING ELSE OUTSIDE THIS FORMAT:

Company Name: [name of the company]\n
Quarter: [ earnings call quarter]\n
Conference call date: [ date of conference call for the transcript]\n
Number of pages in trancript:
\nManagement info: [company managment: names and designation as columns in markdown format ONLY. ]

"""
PROMT_TEMPLATE_CHATBOT = """
You will be given a context of the conversation made so far followed by a customer's question, 
give the answer to the question using the context. 
The answer should be short, straight and to the point. If you don't know the answer, reply that the answer is not available.
Never Hallucinate

Context: {context}

Question: {question}
Answer:
"""

llm = OpenAI(temperature=0.7,openai_api_key=openai_api_key)

# prepping the db 
embedding_function =OpenAIEmbeddings(api_key=openai_api_key)



def main():

    # sidebar navigation
    page_options = ["Welcome","Opening Remarks Summary","Question Answer summary","Chatbot"]
    selected_page = st.sidebar.radio("page options",page_options,label_visibility='collapsed')

    if selected_page == "Welcome":
        # page content based on selection
        st.header("Summarise transcript and download")
        st.subheader("This app will help you summarise earning call transcript.")
        pdf_file = st.file_uploader("Upload your earnings call transcript here and click on 'process'",type=['pdf'])
        button = st.button("Process")
        

        if 'reply' in st.session_state :
            reply = st.session_state.reply
            st.write(reply) 
        elif button:
            file_path = os.path.join("tempdir",pdf_file.name)
            with open(file_path,"wb") as f:
                f.write(pdf_file.getbuffer())
            create_database.generate_data_store(pdf_file=file_path)
            
            db = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)
            query_metadata = "Company name, financial quarter earnings call, conference call date, number of pages in transcript, company management info."

            # search db for relevant context chunks
            results = db.similarity_search_with_relevance_scores(query_metadata,k=5)

            context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])

            prompt = PromptTemplate(template=PROMPT_TEMPLATE_WELCOME,input_variables=["context","query"])

            llm_chain = LLMChain(prompt=prompt,llm=llm)

            bot_reply = llm_chain.run(context=context_text)

            st.write(bot_reply) 
            st.session_state.reply = bot_reply


        
            

    # summarise opening remarks before the QnA in the transcript.
    if selected_page == "Opening Remarks Summary":
        pass

    # summarise the QnA part of the earnings call.
    if selected_page == "Question Answer Summary":
        pass

    # chatbot for entire earning call transcript
    if selected_page == "Chatbot":

        st.header("Chatbot on transcript content :robot_face: ")
        st.text('You can ask questions to chatbot about the content from the transcript,\nit also remembers previous chat history from the current session.')
        db = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)

        def initialize_session_state():

            # Initialize a session state to track whether the initial message has been sent
            if "initial_message_sent" not in st.session_state:
                st.session_state.initial_message_sent = False

            # Initialize a session state to store the input field value
            if "input_value" not in st.session_state:
                st.session_state.input_value = ""

            if "history" not in st.session_state:
                st.session_state.history = []  

            if "chain" not in st.session_state:
                PROMPT = PromptTemplate(
                    template=PROMT_TEMPLATE_CHATBOT, input_variables=['context','question']
                )

                chain_type_kwargs = {"prompt" : PROMPT}

                template = (
                    """Combine the chat history and follow up question into 
                    a standalone question. 
                    If chat hsitory is empty, use the follow up question as it is.
                    Chat History: {chat_history}
                    Follow up question: {question}"""
                    )

                prompt = PromptTemplate.from_template(template)

                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    chain_type="stuff",
                    memory = ConversationBufferWindowMemory(llm = llm,k=5, memory_key='chat_history',input_key='question',output_key='answer',return_messages=True),
                    retriever = db.as_retriever(),
                    condense_question_prompt=prompt,
                    return_source_documents =False,
                    combine_docs_chain_kwargs=chain_type_kwargs,
                )

        def on_click_callback():

            customer_prompt = st.session_state.customer_prompt

            if customer_prompt:
                
                st.session_state.input_value = ""
                st.session_state.initial_message_sent = True

                with st.spinner('Generating response...'):

                    llm_response = st.session_state.chain(
                        {"context": st.session_state.chain.memory.buffer, "question": customer_prompt}, return_only_outputs=True)
                    
                

            st.session_state.history.append(
                Message("customer", customer_prompt)
            )
            st.session_state.history.append(
                Message("AI", llm_response)
            )

        @dataclass
        class Message :
            """Class for keepiong track of chat Message."""
            origin : Literal["Customer","bot"]
            Message : "str"


        initialize_session_state()
        # chat_placeholder = st.container()
        prompt_placeholder = st.form("chat-form")

        with st.form(key="chat_form"):
            # cols = st.columns((6, 1))
            
            # Display the initial message if it hasn't been sent yet
            if not st.session_state.initial_message_sent:
                st.text_input(
                    "Chat",
                    placeholder="Hello, how can I assist you?",
                    label_visibility="collapsed",
                    key="customer_prompt",
                )  
            else:
                st.text_input(
                    "Chat",
                    placeholder="",
                    value=st.session_state.input_value,
                    label_visibility="collapsed",
                    key="customer_prompt",
                )

            st.form_submit_button(
                "Ask",
                type="secondary",
                on_click=on_click_callback,
            )

        # with chat_placeholder:
        for chat in st.session_state.history:
            if type(chat.Message) is dict:
                msg = chat.Message['answer']
            else:
                msg = chat.Message 

            if chat.origin == 'AI': 
                st.write(':robot_face:',msg)
            else:
                st.write(':smiley:',msg)
                # div = f"""
                # <div class = "chatRow 
                # {'' if chat.origin == 'AI' else 'rowReverse'}">
                #     <div class = "chatBubble" {msg}</div>
                # </div>"""
                # st.markdown(div, unsafe_allow_html=True)

        

            

        # Update the session state variable when the input field changes
        st.session_state.input_value = st.text_input




if __name__ == "__main__":
    main()