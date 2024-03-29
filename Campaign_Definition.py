import streamlit as st
import os


# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from streamlit_chat import message




# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI

##langchain packages
from langchain_core.messages import HumanMessage, SystemMessage

import pandas as pd
from PIL import Image
import ast
from ast import literal_eval
from pathlib import Path


openai_api_key =  st.secrets["OPENAI_API_KEY"]





def generate_campaign_template(id=id):
    campaign_schema = Object(
        id=id,
        description="Information about a campaign launch",
        
        # Notice I put multiple fields to pull out different attributes
        attributes=[
            Text(
                id="campaign_name",
                description="The name of the campaign."
            ),
            Text(
                id="target_customers",
                description="The customers for whom the campaign is designed"
            ),
            Number(
                id="campaign_objective",
                description="Metric on which campaign performance is measured"
            )
        ],
        examples=[
            (
                "Create a laptop campaign for computer science students to increase sales",
                [
                    {"campaign_name": "laptop", "target_customers": "computer science students", "campaign_objective":"increase sales"},
                ],
            )
        ]
    )
    return campaign_schema

def generate_target_users_template(id=id):
    target_users_schema = Object(
        id=id,
        description="Information on the target users",
        
        # Notice I put multiple fields to pull out different attributes
        attributes=[
            Number(
                id="lower_end_age",
                description="The lower end of age"
            ),
            Number(
                id="high_end_age",
                description="The higher end of age"
            ),
            Number(
                id="low_purchase_month",
                description="The lower end of purchase month"
            ),
            Number(
                id="high_purchase_month",
                description="The higher end of purchase month"
            ),
            Number(
                id="low_purchase_amount",
                description="The lower end of purchase amount"
            ),
            Number(
                id="high_purchase_amount",
                description="The higher end of purchase amount"
            ),
            Text(
                id="category_purchased",
                description="mentioned purchased category"
            ),
            Text(
                id="item_purchased",
                description="mentioned purchased item"
            ),
            Text(
                id="gender",
                description="mentioned gender"
            ),
            Text(
                id="location",
                description="mentioned location"
            ),
            Text(
                id="season",
                description="mentioned season"
            ),
            Text(
                id="subscription",
                description="mentioned subscription status"
            )
        ],
        examples=[
            (
                "Select users with 23-30 age group and have purchased footwear",
                [
                    {"lower_end_age": 25, "high_end_age": 30, "category_purchased":"footwear"},
                ],  
            ),
            (
                "Select users with age above 50 group in winter season at Oregon location who purchased jeans",
                [
                    {"lower_end_age": 50, "high_end_age": -1, "season":"winter", "location": "Oregon", "item_purchased":"jeans"},
                ],  
            ),
            (
                "Select female users with purchase amount between 50 usd to 100 usd in the month of may",
                [
                    {"gender":"female", "high_purchase_amount": 100, "low_purchase_amount": 50, "high_purchase_month": 5, "low_purchase_month":5},
                ],  
            ),
            (
                "Select subscribed users with purchase amount above 60 usd after June with age below 30",
                [
                    {"subscription": "subscribed", "lower_end_age": -1, "high_end_age": 30,  "high_purchase_amount": -1, "low_purchase_amount": 60, "high_purchase_month": -1, "low_purchase_month":6},
                ],  
            ),
        ]
    )
    return target_users_schema

def get_llm(temperature,max_gen_tokens, selected_model):
    if selected_model == 'GPT 3.5 Turbo':
        selected_model_id = 'gpt-3.5-turbo'
    else:
        selected_model_id = 'gpt-3.5-turbo'
    return ChatOpenAI(
        model_name=selected_model_id,
        temperature=temperature,#0,
        max_tokens=max_gen_tokens,#2000,
        openai_api_key=openai_api_key)
        

# def save_data_to_Chroma(docs):
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma.from_documents(docs, embeddings)
#     return db

def retrieve_camp_definition(text, llm):
    text=text
    id='campaign'
    schema=generate_campaign_template(id)
    chain = create_extraction_chain(llm, schema)
    output = chain.predict_and_parse(text=text)['data'][id]['data'][id][0]
    return output

def retrieve_target_users(text, llm):
    text=text
    id='target_users'
    schema=generate_target_users_template(id)
    chain = create_extraction_chain(llm, schema)
    output = chain.predict_and_parse(text=text)['data'][id]['data'][id][0]
    return output

def select_relevant_attr(df,key_attributes,llm):
    all_columns=list(df.columns)
    messages = [
        SystemMessage(
            content="You are a helpful assistant that returns relevant words output as a dictionary"
        ),
        HumanMessage(
            content=f'What are the closest relevant words for the words in {key_attributes} in the following list of words {all_columns}'
        ),
    ]
    llm_response=llm.invoke(messages)
    return literal_eval(llm_response.content)


def main():
    icon = Image.open('letter-c.png')
    text_file_path='selected_attr.txt'
    csv_file_path='req_data.csv'
    camp_def_file_path='camp_def.txt'

    st.set_page_config(page_title='CampAI', page_icon=icon, layout='wide')
    st.title('CampAIgn Automation Tool')

    

    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'
        st.session_state['uploaded_file']='req_data.csv'

        st.session_state['max_gen_length'] = 2000
        # st.session_state['top_p'] = 0.5
        st.session_state['temp'] = 0.0

        st.session_state['process_status_camp_def'] = False
        st.session_state['process_status_relevant_attr'] = False
        st.session_state['process_status_user_filter'] = False

        # st.session_state['download_header'] = None

        st.session_state['llm'] = None
        st.session_state['processed_campaign_def'] = None
        st.session_state['process_relevant_attr']=None
        # st.session_state['vector_db'] = None
        st.session_state['selected_model'] = 'GPT 3.5 Turbo'


    if st.session_state['home_option']=='initialized':
        st.subheader('Upload Document')
        uploaded_file = st.file_uploader("Upload your document (CSV, XLSX)", type=['csv', 'xlsx'])


        if uploaded_file is not None:

            st.subheader('Uploaded File')
            _, file_extension = os.path.splitext(uploaded_file.name)
            if file_extension.lower()=='.csv':
                data= pd.read_csv(uploaded_file.name,index_col=0)
                data.to_csv('req_data.csv')
                st.dataframe(data.head())
            elif  file_extension.lower()=='.xlsx':
                data= pd.read_excel(uploaded_file.name,index_col=0)
                data.to_csv('req_data.csv')
                st.dataframe(data.head())

            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['process_status_relevant_attr']=None

            
            llm = get_llm(st.session_state['temp'],st.session_state['max_gen_length'], st.session_state['selected_model'])
            st.session_state['llm'] = llm
        
            st.subheader('Campaign Definition')
            campaign_def=st.text_area("", value="", height=20, max_chars=200, placeholder="Enter your campaign definition", disabled=False, label_visibility="visible")
            enter_button=st.button("Enter", key='cam_def', help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)
        

            if enter_button and len(campaign_def)>0:

                st.session_state['process_status_relevant_attr']=None
                
                with st.spinner("Processing..."):

                    campaign_def_output = retrieve_camp_definition(campaign_def, llm)

                    with open(camp_def_file_path, 'w') as file:
                        file.write(str(campaign_def_output))

                    key_attributes=list(campaign_def_output.values())
                    st.session_state['processed_campaign_def'] = campaign_def_output
                    
                    process_status_camp_def = st.success(f"Following key attributes derived from your campaign definition")
                    st.session_state['process_status_camp_def'] = process_status_camp_def  

                    st.dataframe(pd.DataFrame(campaign_def_output.items(),columns=['key attributes','Retreived value']))
                
                if st.session_state['process_status_camp_def']:
                    
                    relevant_attributes_output=select_relevant_attr(data,key_attributes,llm)

                    selected_attributes=[]
                    for item in relevant_attributes_output.values():
                        selected_attributes.extend(item)

                    st.session_state['process_relevant_attr'] =relevant_attributes_output

                    process_status_relevant_attr = st.success(f"Following relevant attributes derived from your dataset based on campaign definition")
                    st.session_state['process_status_relevant_attr'] = process_status_relevant_attr

                    st.dataframe(pd.DataFrame(relevant_attributes_output.items(),columns=['Retreived value','Relevant attributes']))

                    with open(text_file_path, 'w') as file:
                        # Join the list elements into a single string with a newline character
                        data_to_write = '\n'.join(selected_attributes)
                        # Write the data to the file
                        file.write(data_to_write)

                else:
                    
                    st.error("Processing campaign Definition Failed, Please Try again!")

            
            elif enter_button:
                st.session_state['process_status_relevant_attr']=None
                st.error('Please mention campaign definition')

        elif Path(csv_file_path).exists():
            data= pd.read_csv(csv_file_path,index_col=0)
            st.subheader('Uploaded File')
            st.dataframe(data.head())
            try:
                if st.session_state['process_status_relevant_attr']:
            
                    st.success(f"Following key attributes derived from your campaign definition")

                    st.dataframe(pd.DataFrame( st.session_state['processed_campaign_def'].items(),columns=['key attributes','Retreived value']))

                    st.success(f"Following relevant attributes derived from your dataset based on campaign definition")

                    st.dataframe(pd.DataFrame(st.session_state['process_relevant_attr'].items(),columns=['Retreived value','Relevant attributes']))
            except:
                st.write('Above is the selected dataframe')
        
        else:
                st.error('Please upload a file to get started.')

            
       
        

       

    
if __name__ == "__main__":
    main()
