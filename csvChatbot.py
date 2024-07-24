import streamlit as st
import pandas as pd
import os
import plotly.express as px

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.agents import AgentType

os.environ["GOOGLE_API_KEY"] = "AIzaSyDgec0Y__YGL5FEz7doYq9hlpIgRusq8Yg"



# Set page title
st.set_page_config(page_title='LCT chatbot')
st.title('Lct Chatbot')
# display a logo in the app
st.image('BI_Logo.png', width=200)

# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        #st.write(df)
        df_melted = df.melt(id_vars='Year', value_vars=['Malaysia', 'Thailand', 'Japan', 'China'], 
                    var_name='Country', value_name='Value')

        # Creating the stacked bar chart
        fig = px.bar(df_melted, x='Year', y='Value', color='Country', 
                    title='Value for Each Country Every Year',
                    labels={'Value': 'Value', 'Year': 'Year', 'Country': 'Country'},
                    text='Value')

        # Update layout for better visualization
        fig.update_layout(barmode='stack', 
                          xaxis_title='Year', 
                          yaxis_title='Value',
                          legend_title='Country')

        # Adding the values on top of the bars
        fig.update_traces(texttemplate='%{text:.2s}', textposition='none')

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)

    return df

# Generate LLM response
def generate_response(csv_file, input_query):
    # title_template = PromptTemplate(input_variables=['topic'],
    #                                 template='Give me medium article title on {topic}')
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    # other params...
  )
    df = load_csv(csv_file)
    
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(llm,
                                          df, 
                                          verbose=True, 
                                          agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                          allow_dangerous_code=True,
                                          handle_parsing_errors=True)
    
    # Perform Query using the Agent
    response = agent.run(input_query)
    return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])


# App logic

query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)
if uploaded_file is not None and query_text:
    st.header('Output')
    generate_response(uploaded_file, query_text)
