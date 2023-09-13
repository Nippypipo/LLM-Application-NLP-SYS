import os
from apikey import apikey

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory


os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🤡 Parody Literature Creation 📝')
prompt_title = st.text_input('Plug in your title here')
prompt_genre = st.selectbox(
    'Select your genre',
    ('Action-Adventure', 'Drama', 'Comedy', 'Science fiction','Romance','Western','Horror'))
prompt_moral = st.text_input('Plug in your moral here')
submit_button = st.button("Submit")

# Prompt templates
story_template = PromptTemplate(
    input_variables = ['title', 'genre', 'moral'],
    template=
    """create a story of {title} in simple language, keep the main original plot, change the genre to {genre} 
    , and add the moral lesson about {moral}. The response needs to be less than 200 words and starts with the title of the story 
    and then ends with a summary of the moral lesson. 
using these information as your guidance: 
1. Definitions of genres: 
1.1 Comedy: Humorous stories that are intended to make the reader laugh by using jokes. 
1.2 Horror: Stories that are intended to create feelings of fear, dread, and terror. 
1.3 Action-adventure: Stories that feature physical danger, thrilling near misses, and courageous feats. 
1.4 Science fiction: Stories that feature imagined elements that are inspired by science or social science. 
1.5 Romance: Stories that focus on a love story between two people. 
1.6 Western: Stories that tell the tale of a cowboy or gunslinger pursuing an outlaw in the Wild West. 
1.7 Drama: Stories that feature high stakes, many conflicts, and emotionally-driven characters. 
2. Moral lesson: A value that is acquired through reflection and understanding of a story or a specific life situation.
"""
)

dialogue_template = PromptTemplate(
    input_variables = ['story'],
    template = "Write an example dialogue less than 200 words that show the moral lesson of this story STORY: {story}"
)

# Memory
story_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
dialogue_memory = ConversationBufferMemory(input_key='story', memory_key='chat_history')

# LLMS
llm = ChatOpenAI(temperature= 0.5)
story_chain = LLMChain(llm=llm, prompt=story_template, verbose=True
                       , output_key='story', memory=story_memory)
dialogue_chain = LLMChain(llm=llm, prompt=dialogue_template, verbose=True
                        , output_key='dialogue', memory=dialogue_memory)
sequential_chain = SequentialChain(chains=[story_chain, dialogue_chain]
                                   , input_variables=['title', 'genre', 'moral']
                                   , output_variables=['story', 'dialogue'], verbose = True)

# Show stuff to the screen if there's prompt
if prompt_title and prompt_genre and prompt_moral:
    response = sequential_chain({'title':prompt_title,'genre':prompt_genre, 'moral':prompt_moral})
    st.write(response['story'])
    st.write(response['dialogue'])

    with st.expander('Story History'):
        st.info(story_memory.buffer)
    with st.expander('Dialogue History'):
        st.info(dialogue_memory.buffer)


        
