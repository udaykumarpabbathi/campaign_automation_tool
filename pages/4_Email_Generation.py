import streamlit as st
import pandas as pd
from PIL import Image
from ast import literal_eval

# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

##langchain packages
from langchain_core.messages import HumanMessage, SystemMessage


openai_api_key = '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'

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

def generate_email(llm, camp_def_dict, selected_cluster_descr):
    campaign_name=camp_def_dict['campaign_name']
    target_users=camp_def_dict['target_customers']
    campaign_objective=camp_def_dict['campaign_objective']

    messages = [
        SystemMessage(
            content="You are a helpful marketing analyst that returns a email body"
        ),
        HumanMessage(
            content=f'writes an email campaign on {campaign_name} for {target_users} to {campaign_objective} who are part of a customer segment and this cluster consists of {selected_cluster_descr}'
        ),
    ]
    llm_response=llm.invoke(messages)
    return llm_response.content

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def main():
    icon = Image.open('letter-c.png')
    st.set_page_config(page_title='CampAI', page_icon=icon, layout='wide')
    st.title('CampAIgn Automation Tool')
    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'

        st.session_state['max_gen_length'] = 2000
        st.session_state['temp'] = 0.0
        st.session_state['llm'] = None
        st.session_state['selected_model'] = 'GPT 3.5 Turbo'


        st.session_state['process_status_email_gen'] = False
        st.session_state['process_email_gen']=None
    
    if st.session_state['home_option']=='initialized':
        st.subheader('Generate Email campaign')
        try:
            cluster_descriptions=pd.read_csv('cluster_descriptions.csv',index_col=0)

            cluster_names=[col for col in cluster_descriptions.columns if 'cluster' in col.lower()]
            cluster_option = st.selectbox('Select the cluster',cluster_names)
            selected_cluster_descr=cluster_descriptions[cluster_option]['description']
            st.write("Cluster description:",selected_cluster_descr)

            enter_button=st.button("Generate Email", key='email_gen', help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)

            if enter_button:
                cluster_descriptions=pd.read_csv('cluster_descriptions.csv',index_col=0)
                
                
                with st.spinner(" Generating Email content..."):

                    llm = get_llm(st.session_state['temp'],st.session_state['max_gen_length'], st.session_state['selected_model'])

                    txt_file = open('camp_def.txt', 'r')
                    camp_def_text = txt_file.read()
                    camp_def_dict=literal_eval(camp_def_text)
                    
                    email_content=generate_email(llm, camp_def_dict, selected_cluster_descr)

                    process_status_email_gen = st.success(f"Generated campaign Email Content for {cluster_option}")
                    st.session_state['process_status_email_gen']=process_status_email_gen

                    if st.session_state['process_status_segmentation']:
                        st.session_state['process_email_gen']=email_content
                        # st.write(email_content)
                        st.text_area("Email:\n", value=email_content, height=500)
                    
                        enter_button_2=st.button("Send Email", key='target_users_email', args=None,  on_click=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)
                            

            else:
                try:
                    if st.session_state['process_status_email_gen']:
                        st.success(f"Generated campaign Email Content for the selected Cluster")
                        st.text_area("Email:\n", value=st.session_state['process_email_gen'],height=500)
                except:
                    st.error('Click Generate Email to get started')
        
        except:
                st.error("Finish segmentation section before email generation")

        


if __name__ == "__main__":
    main()

