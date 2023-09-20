import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory


os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]

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

wordplay_list_template = PromptTemplate(
    input_variables = ['title'],
    template = 
    """ Give me 6 pairs of phrases that have the pattern of similarity that maintain some phonetic similarity where the first word and the last word have exchanged their first consonants like these example in theree slash:
    ///
    - "cut my hair" and "care my heart"
    - "bad guy" and "buy gas".
    - "hard disk" and "this heart"
    - "pat my back" and "bat my pack"
    - "lit my candle" and "kit my handle"
    - "cap my bottle" and "bap my cottle"
    - "hug my bear" and "bug my hair"
    - "tack my board" and "back my toard"
    ///
    
    and the response should follow the instructions below.

1. all pair of phrases should have the real meaning in English language
2. all pair of phrases should be relevant to {title} story
    """
)

wordplay_dialogue_template = PromptTemplate(
    input_variables = ['wordplay'],
    template = 
    """
    Use some list in the LISTS that contains 10 wordplay pairs of phrases to create a dialogue where the phrases in the same list should be in the same sentence in dialogue.
    You should mark asterisk sign between the pharses from the pairs of phrases that you chose from the LISTS in the dialogue that you create.
    For example, each phrases in particular list should be in the same sentence.
    LISTS: '''{wordplay}'''

    For example, if you choose the pairs of pharses from this three list
    - "cut my hair" and "care my heart"
    - "bad guy" and "buy gas".
    - "hard disk" and "this heart"
    The response should be like the example text in the three slash.

    ///
    characterA: I want to "cut my hair" because you do not "care my heart". 

    characterB: He is not a "bad guy", he just "buy gas" 

    characterA: I want to remove him from the "hard disk", but he is still in "this heart". 
    ///

"""
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
wordplay_list_chain = LLMChain(llm=llm, prompt=wordplay_list_template, verbose=True
                        , output_key='wordplay')
wordplay_dialogue_chain = LLMChain(llm=llm, prompt=wordplay_dialogue_template, verbose=True
                        , output_key='dialogue_wordplay')

sequential_chain = SequentialChain(chains=[story_chain, dialogue_chain, wordplay_list_chain, wordplay_dialogue_chain]
                                   , input_variables=['title', 'genre', 'moral']
                                   , output_variables=['story', 'dialogue', 'wordplay', 'dialogue_wordplay'], verbose = True)

# Show stuff to the screen if there's prompt
if prompt_title and prompt_genre and prompt_moral:
    response = sequential_chain({'title':prompt_title,'genre':prompt_genre, 'moral':prompt_moral})
    st.write(response['story'])
    st.write(response['dialogue'])
    st.write(response['wordplay'])
    st.write(response['dialogue_wordplay'])

    with st.expander('Story History'):
        st.info(story_memory.buffer)
    with st.expander('Dialogue History'):
        st.info(dialogue_memory.buffer)



