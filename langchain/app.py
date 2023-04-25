import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from langchain.memory import ConversationBufferMemory #add memory 

from langchain.utilities import WikipediaAPIWrapper  #langchain tools

os.environ['OPEN_APPI_KEY'] = apikey
#App framework - step 1
st.title("Youtube GPT creator")
prompt = st.text_input('Plug in your prompt')

#Prompt Template  step -4
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = "write me a youtube video title about {topic}"
)

# SCript Template  step -7 -- Updated
script_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template = "write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research}"
)

#Memory - for History - step 12 - not for prompting
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')


#LLms -- step 2
llm = OpenAI(temperature=0.9)
#step 5
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True,output_key="title",memory=title_memory)
# script chain step 8
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key= 'script',memory=script_memory)

# Step 16 - Delete it# #sequential chain step 9
# sequential_chain=SequentialChain(chains = [title_chain,script_chain],input_variables=['title'],output_variables=['title','script'],verbose=True)

# step17
wiki = WikipediaAPIWrapper()
#show stuff to the screen if there's a prompt -- step 3
if prompt :
    # response = llm(prompt)
    # response = title_chain.run(topic=prompt)    #step 6
    # response = sequential_chain.run(prompt)    #step 10
    # response = sequential_chain({'topic':prompt})    #step 11
    # --step 18--
    title =title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)
    # st.write(response['title'])
    st.write(title)
    # st.write(response['script'])
    st.write(script)
    # st.write(response)

    #Render to screeen - Mempry step 14
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
