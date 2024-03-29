import streamlit as st
import pandas as pd
from PIL import Image

# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number, Bool

# LangChain Models
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI


openai_api_key = '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'



def generate_target_users_template(id=id):
    target_users_schema = Object(
        id=id,
        description="Information on the target users",
        
        attributes=[
            Number(
                id="lower_end_age",
                description="The lower end of age"
            ),
            Number(
                id="high_end_age",
                description="The higher end of age"
            ),
            Text(
                id="category_purchased",
                description="mentioned purchased category"
            ),
            Text(
                id="gender",
                description="mentioned gender"
            ),
            Bool(
                id="email_click",
                description="if clicked an email"
            ),
            Bool(
                id="email_open",
                description="if opened an email"
            ),
            Bool(
                id="purchased",
                description="if purchased a category"
            )
        ],
        examples=[
            (
                "Select users with 23-30 age group and have purchased footwear",
                [
                    {"lower_end_age": 25, "high_end_age": 30, "category_purchased":"footwear", "purchased": True},
                ],  
            ),
            (
                "Select female users with age above 50 and opened email and not purchased",
                [
                    {"lower_end_age": 50, "high_end_age": -1, "gender": 'female', "email_open": True, "purchased": False},
                ],  
            ),
            (
                "Select users who have not purchased but clicked and opened emails",
                [
                    {"purchased": False, "email_click": True,  "email_open": True },
                ],  
            ),
            (
                "Select users who have clicked email but not opened with age below 30 and purchased accessories",
                [
                    {"email_click": True, "email_open": False, "lower_end_age": -1, "high_end_age": 30, "category_purchased":"accessories", "purchased": True },
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
        


def retrieve_target_users(text, llm):
    text=text
    id='target_users'
    schema=generate_target_users_template(id)
    chain = create_extraction_chain(llm, schema)
    output = chain.predict_and_parse(text=text)['data'][id]['data'][id][0]
    return output


def filter_dataset(selected_attributes,data,target_user_dict):
    for key in list(target_user_dict.keys()):
        if key=='lower_end_age':
            data=data[data.Age>int(target_user_dict[key])]
            if 'Age' not in selected_attributes:
                selected_attributes.append('Age')
        if key=='high_end_age':
            data=data[data.Age<int(target_user_dict[key])]
            if 'Age' not in selected_attributes:
                selected_attributes.append('Age')
        if key=='category_purchased':
            data=data[data.Category==target_user_dict[key]]
            if 'Category' not in selected_attributes:
                selected_attributes.append('Category')
        if key=='email_click':
            if target_user_dict[key]:
                data=data[data.email_engagement_status!='not_click']  
            else:
                data=data[data.email_engagement_status=='not_click']

            if 'email_engagement_status' not in selected_attributes:
                selected_attributes.append('email_engagement_status')
        if key=='email_open':
            if target_user_dict[key]:
                data=data[data.email_engagement_status!='not_open']  
            else:
                data=data[data.email_engagement_status=='not_open']

            if 'email_engagement_status' not in selected_attributes:
                selected_attributes.append('email_engagement_status')

        if key=='purchased':
            if target_user_dict[key]!=False:
                data=data[data.activation_status!='purchased']
            if 'activation_status' not in selected_attributes:
                selected_attributes.append('activation_status')

    return selected_attributes,data


def main():
    icon = Image.open('letter-c.png')
    st.set_page_config(page_title='IMPACT.ai', page_icon=icon, layout='wide')
    st.title('IMPACT.ai')
    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'

        st.session_state['max_gen_length'] = 2000
        st.session_state['temp'] = 0.0
        st.session_state['llm'] = None
        st.session_state['selected_model'] = 'GPT 3.5 Turbo'


        st.session_state['process_status_user_filter'] = False
        st.session_state['processed_data_filter']=False
        

    if st.session_state['home_option']=='initialized':
        # Code for option 2

        llm = get_llm(0.0,2000, 'GPT 3.5 Turbo')
        
        st.subheader('Define Targeted Users')
        user_filter=st.text_area("", value="", height=20, max_chars=200, placeholder="Enter conditions for your target users", disabled=False, label_visibility="visible")
        enter_button_2=st.button("Enter", key='target_users', help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)


        if  enter_button_2 and len(user_filter)>0:

            st.session_state['process_status_user_filter']=None

            with st.spinner("Processing..."):
                user_filter_output = retrieve_target_users(user_filter, llm)

                user_filter_values=user_filter_output.copy()
                for key,value in user_filter_values.items():
                    if not value:
                        del user_filter_output[key]


                process_status_user_filter = st.success(f"Following attributes derived from given user filter")
                st.session_state['process_status_user_filter'] = process_status_user_filter
            
                st.session_state['processed_user_filter'] = user_filter_output
                st.dataframe(pd.DataFrame(user_filter_output.items(),columns=['User filter','Retreived value']))

                try:
                    data=pd.read_csv('req_data.csv',index_col=0)

                    object_columns=data.select_dtypes(include=[object]).columns
                    for col in object_columns:
                        data[col]=data[col].apply(str.lower)
                    
                    txt_file = open('selected_attr.txt', 'r')
                    selected_attributes_txt = txt_file.read() 

                    selected_attributes = selected_attributes_txt.split("\n") 

                    final_selected_attributes,final_data=filter_dataset(selected_attributes,data,user_filter_output)
                    final_data=final_data[final_selected_attributes]

                    st.success(f"Filtered dataset based on above dictionary pairs and downloaded users to final_data.csv")
                    st.session_state['processed_data_filter'] = final_data
                    st.dataframe(final_data.head())

                    final_data.to_csv('final_data.csv')
                
                except:
                    st.error('Finish campaign definition section before filtering on above attributes')


        elif enter_button_2:
            st.session_state['process_status_user_filter']=None
            st.error('Please mention target users')
        
        else:
            try:
                if st.session_state['process_status_user_filter']:
                    st.success(f"Following attributes derived from given user filter")
                    st.dataframe(pd.DataFrame(st.session_state['processed_user_filter'].items(),columns=['User filter','Retreived value']))

                    if st.session_state['processed_data_filter'] is not None:
                        st.success(f"Filtered dataset based on above dictionary pairs")
                        st.dataframe(st.session_state['processed_data_filter'].head())
            except:
                st.error('Enter details to process targeted users')

    
if __name__ == "__main__":
    main()
