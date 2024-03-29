import streamlit as st
import pandas as pd
from PIL import Image
import os



def main():
    icon = Image.open('letter-c.png')
    st.set_page_config(page_title='CampAI', page_icon=icon, layout='wide')
    st.title('CampAIgn Automation Tool')
    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'
    
    if st.session_state['home_option']=='initialized':
        st.subheader('Campaign Performance Measure')
        uploaded_file = st.file_uploader("Upload Campaign Performance Document (CSV, XLSX)", type=['csv', 'xlsx'])

        if uploaded_file is not None:

            st.subheader('Uploaded File')
            _, file_extension = os.path.splitext(uploaded_file.name)
            if file_extension.lower()=='.csv':
                new= pd.read_csv(uploaded_file.name,index_col=0)
                st.dataframe(new.head())
            elif  file_extension.lower()=='.xlsx':
                new= pd.read_excel(uploaded_file.name,index_col=0)
                st.dataframe(new.head())
            try:
                old=pd.read_excel('data_campaign_1_results.xlsx',index_col=0)
                # new=pd.read_csv('campaign2_dummy_data.csv')
                with st.spinner("Measuring Performance..."):
                    dict_cross_df={}
                    dict_metrics_df={}
                    for cluster in new.Cluster:
                        dict_cross_df[cluster]=pd.crosstab(new[new.Cluster==cluster]['activation_status'], [old['activation_status']], rownames=['campaign_2'], colnames=['campaign_1']) 
                        
                        new_open_rate = len(new[(new.Cluster==cluster) & (new['email_engagement_status']!='not_open')])/len(new)
                    

                        new_click_rate = len(new[(new.Cluster==cluster) & (new['email_engagement_status']=='clicked')])/len(new)
                    

                        new_puchase_rate = len(new[(new.Cluster==cluster) & (new['activation_status']=='purchased')])/len(new[(new.Cluster==cluster) & (new['email_engagement_status']=='clicked')])

                        dict_metrics_df[cluster]={'open_rate':new_open_rate*100,
                                                'click_rate': new_click_rate*100,
                                                'purchase_rate':new_puchase_rate*100}
                    
                    st.subheader('Cluster 0')

                    st.dataframe(dict_cross_df[0])
                    st.dataframe(dict_metrics_df[0])

                    st.subheader('Cluster 1')

                    st.dataframe(dict_cross_df[1])
                    st.dataframe(dict_metrics_df[1])

                    st.subheader('Cluster 2')

                    st.dataframe(dict_cross_df[2])
                    st.dataframe(dict_metrics_df[2])

                    st.subheader('Cluster 3')

                    st.dataframe(dict_cross_df[3])
                    st.dataframe(dict_metrics_df[3])

                    
                
            except:
                    st.error("Could not process...")



if __name__ == "__main__":
    main()