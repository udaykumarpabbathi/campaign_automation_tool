import streamlit as st

from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import ast
from ast import literal_eval

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import PromptTemplate

openai_api_key = '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'



def generate_prompt_template():
    template = """
        System: Use the following pieces of context to answer the users question.  
        Create user profiles for each 'Cluster' type of users.
        Each data point is a transaction done by a customer to buy some product and people have been assigned in clusters based on these information. The product can be found from category column where we have broad to narrow category of the product. We also have the purchase amount of the item and some person details like his age, others (the column names are self descriptory). 
        Based on all of these information for each assigned cluster, create the profile of each cluster. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}

        Answer: 
        """
    return PromptTemplate(template=template, input_variables=['context','question'])

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

def generate_cluster_description(llm, PROMPT, db):
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 chain_type_kwargs={"prompt": PROMPT},retriever=db.as_retriever(),return_source_documents=True,verbose = True)
    
    txt_file = open('camp_def.txt', 'r')
    camp_def_text = txt_file.read() 

    question = f"Give a name and description to all the clusters 0 to 3 based on {camp_def_text} in the dictionary format with string values. Also do add information as you deem appropriate."

    response = qa({"query":question})

    return response['result']

def main():
    icon = Image.open('letter-c.png')
    st.set_page_config(page_title='IMPACT.ai', page_icon=icon, layout='wide')
    st.title('IMPACT.ai')

    cluster_data_path='final_cluster_data.csv'
    n_clusters=4#cluster
    chunk_size=200##embeding
    model_name="all-MiniLM-L6-v2"#sentence transformer

    
    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'
        st.session_state['process_status_segmentation']=False
        
        st.session_state['process_status_cluster_descr']=False

        st.session_state['max_gen_length'] = 2000
        st.session_state['temp'] = 0.0
        st.session_state['llm'] = None
        st.session_state['selected_model'] = 'GPT 3.5 Turbo'

    
    if st.session_state['home_option']=='initialized':

        enter_button=st.button("Process Segmentation", key='seg', help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)

        if enter_button:
            try:
                final_data=pd.read_csv('final_data.csv',index_col=0)
                object_columns=final_data.select_dtypes(include=[object]).columns
                cluster_data=final_data.copy()
                for col in object_columns:
                    cluster_data[col] = pd.Categorical(final_data[col]).codes

                best_kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=45)
                final_data['Cluster'] = best_kmeans.fit_predict(cluster_data)

                process_status_segmentation = st.success(f"Segmented Targeted Users into {n_clusters} clusters")
                st.session_state['process_status_segmentation']=process_status_segmentation
                st.dataframe(final_data.head())

                final_data.to_csv('final_cluster_data.csv')

                # enter_button_2=st.button("Generate Cluster Description", key='seg_descr', help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False) 

                if st.session_state['process_status_segmentation']:

                    with st.spinner(" Generating cluster descriptions..."):
                        loader = CSVLoader(cluster_data_path)
                        documents = loader.load()
                        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size)
                        docs = text_splitter.split_documents(documents)

                        embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
                        db = Chroma.from_documents(docs, embedding_function) #docs limit of 41k
                        
                        PROMPT = generate_prompt_template()

                        llm = get_llm(st.session_state['temp'],st.session_state['max_gen_length'], st.session_state['selected_model'])


                        cluster_descriptions=generate_cluster_description(llm, PROMPT, db)

                        process_status_cluster_descr = st.success(f"Following are the cluster descriptions")
                        st.session_state['process_status_cluster_descr']=process_status_cluster_descr

                        st.session_state['process_cluster_descr']=cluster_descriptions
                        # st.write(cluster_descriptions)
                        cluster_descriptions_df=pd.DataFrame(literal_eval(cluster_descriptions))
                        st.dataframe(cluster_descriptions_df)

                        cluster_descriptions_df.to_csv('cluster_descriptions.csv')


            except:
                st.error("Finish target user selection before segmentation")
        
        else:
            try:
                final_data=pd.read_csv(cluster_data_path,index_col=0)
                st.success(f"Segmented Targeted Users into {n_clusters} clusters")
                st.dataframe(final_data.head())

                if st.session_state['process_status_cluster_descr']:
                    # st.success(f"Following are the cluster descriptions")
                    # st.write(st.session_state['process_cluster_descr'])
                    st.dataframe(pd.DataFrame(literal_eval(st.session_state['process_cluster_descr'])))
            except:
                st.error('Click process segmentation for creating clusters')

        # else:
        #     st.session_state['process_status_segmentation']=None




if __name__ == "__main__":
    main()
