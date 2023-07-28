import os 
from langchain.chat_models import ChatAnthropic
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import AIMessage, HumanMessage, SystemMessage


from config.settings import ANTHROPIC_API_KEY
#os.environ["ANTHROPIC_API_KEY"] = ""#os.environ.get('ANTHROPIC_API_KEY')

class ClaudeLLM:
    """
    Utility class to initialise the LLM model as well as pre-process the data corpus,
    then to use as a wrapper for simple queries.
    """

    def __init__(self):
        """
        Initialisae the class, passing in the locations of source data and models to use

        Parameters
        """

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Verbose is required to pass to the callback manager
        # n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
        # n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        # n_ctx = 512 * 2

        self.llm = ChatAnthropic(
            callback_manager=callback_manager,
            anthropic_api_key=ANTHROPIC_API_KEY,
        )
        
        self.setup()

    def setup(self):
        """
        Ingests source data, runs data pre-processing and sets up vector DB
        for embeddings.
        """

        template = """
        You are a mind reader with magical abilities.
        You will be given something to guess, such as an animal, or a famous person.
        You will ask a question, I will provide an answer, and then you will guess.
        If your guess is wrong, then you must ask another question.
        Repeat this until you get the right answer.
        Your goal is to find the right answer in as few questions as possible. Only make a guess
        when you are confident, otherwise ask more questions that narrow down the possibilities.

        {history}
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

        summarise_template= """
        Please provide: high level summary in plain English including relevant context 
        from web searches (bullet pointed) and relevant actions in a project plan table 
        including action holder and timeframes.
        Additionally, please include the following appendices: appendix 1: relevant media 
        coverage including sources, date, summary and relevance to the email.
        Appendix 2: relevant academic articles presented in a table article title, 
        authors, publication year, journal and summary (present appendices in tables) for 
        the following email: {email}

        Summarisation:"""
        summarise_prompt = PromptTemplate(input_variables=["email"], template=summarise_template)

        self.summarise_chain = LLMChain(
            llm=self.llm,
            prompt=summarise_prompt,
            verbose=True,
        )

        self.first_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=10),
        )

        return
    
    def get_chain(self, chain_type = None):
        """ Gets required LLM chain
        
        TODO pass keys for different chains
        """
        if chain_type == "summarise":
            return self.summarise_chain

        return self.first_chain

