import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
import joblib
import re


def main():
    icon = Image.open('letter-c.png')
    st.set_page_config(page_title='CampAI', page_icon=icon, layout='wide')
    st.title('CampAIgn Automation Tool')

    cluster_data_path='final_cluster_data.csv'

    if 'home_option' not in st.session_state:
        st.session_state['home_option'] = 'initialized'
    
    if st.session_state['home_option']=='initialized':
        st.subheader('Campaign Performance Prediction')
        try:
            cluster_descriptions=pd.read_csv('cluster_descriptions.csv',index_col=0)

            cluster_names=[col for col in cluster_descriptions.columns if 'cluster' in col.lower()]
            cluster_option = st.selectbox('Select the cluster',cluster_names)
            selected_cluster_descr=cluster_descriptions[cluster_option]['description']
            st.write("Cluster description:",selected_cluster_descr)

            cluster_value=re.findall(r'\d+', cluster_option)
            final_data=pd.read_csv(cluster_data_path,index_col=0)
            cluster_data=final_data[final_data.Cluster==int(cluster_value[0])]
            cluster_avg_age=cluster_data['Age'].mean()
            cluster_avg_item_cost=cluster_data['Purchase Amount (USD)'].mean()
            cluster_category=cluster_data['Category'].mode()

            # Load the saved model for prediction
            loaded_model = joblib.load('campaign_model.pkl')

            sent_volume=st.slider('Sent Volume', min_value=2000, max_value=4000, value=2500, help="Adjust the campaign sent volume (users)")
            campaign_duration_months=st.slider('Campaign Duration', min_value=1, max_value=6, value=2, help="Adjust the campaign duration(months)")

            # print(cluster_category)
            test_data={
            'avg_Age': {0: cluster_avg_age},
            'category': {0: cluster_category[0]},
            'avg_item_cost': {0: cluster_avg_item_cost},
            'gender': {0: 'all'},
            'geography': {0: 'all'},
            'campaign_duration_months': {0: campaign_duration_months},
            'sent_volume': {0: sent_volume}
            }

            test_data_df=pd.DataFrame.from_dict(test_data)

            st.subheader(f'{cluster_option} attributes: ')
            st.dataframe(test_data_df)

            # predict test data on the model
            prediction = loaded_model.predict(test_data_df)

            st.subheader('Campaign Convertion Rate:')
            st.markdown(f'**{prediction[0]}**.')
        except:
             st.error("Finish segmentation section before performance prediction")
        


if __name__ == "__main__":
    main()
