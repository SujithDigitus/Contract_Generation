import os
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import fitz
# from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain.chains import LLMChain
import torch
from PIL import Image
import numpy as np
import chromadb
# from langchain.retrievers import VectorstoreRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever, )

# DocumentCompressorPipeline


set_debug(True)

import openai
import os

client = OpenAI()
# Define your API key

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import openai
from langchain.retrievers import MultiQueryRetriever

import pprint



def rag_pipeline_with_prompt(context):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Define your prompt template
    template = """
Analyze the following contract and identify all fields that are specific to the parties or the agreement (e.g., party names, dates, amounts, addresses, terms, and agreements). Replace each with a descriptive placeholder in curly braces.
 
Then return a valid JSON object with this structure:
{{
  "Template": "<full contract with placeholders, preserving original formatting and indentation>",
  "Placeholders": {{
    "<PLACEHOLDER_1>": "Short description of what this field represents.",
    "<PLACEHOLDER_2>": "Another description.",
    ...
  }}
}}
Respond ONLY with a valid JSON object, no markdown, no explanations.

Input context:
{context}
"""

    prompt_final = PromptTemplate(
        template=template,
        input_variables=[ "context"]
    )
    rag_chain = (
        {
            "context": itemgetter("context")
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    print(f"context of the query is {context}")
    result = rag_chain.invoke({"context":context})
    return result # for testing image is not included


def get_text_from_Pdf(file_path):
        doc = fitz.open(file_path)
    
    # Initialize a variable to hold the text
        full_text = ""
    
    # Iterate over each page in the document
        for page_num in range(doc.page_count):
        # Get a page
            page = doc.load_page(page_num)
        
        # Extract text from the page
            page_text = page.get_text()
        
        # Append the extracted text to the full_text string
            full_text += page_text
        
    # Close the document
        doc.close()
    
        return full_text

#generation of the contract using user input
def Generation_of_Contract(context,Placeholder):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Define your prompt template
    template = """
You are a document generation assistant. Your task is to generate a final contract by replacing the placeholders in the given template with real values provided by the user.

Here is the template:
{Template}

Here are the user-provided values for each placeholder:
{Placeholder}

Instructions:
- Replace each placeholder in the template (written inside {{Placeholder_Name}}) with its corresponding value.
- Do not change any part of the text other than replacing placeholders.
- Return the full final document only, without any additional explanation. """
    prompt_final = PromptTemplate(
        template=template,
        input_variables=[ "Template","Placeholder"]
    )
    rag_chain = (
        {
            "Template": itemgetter("Template"),
            "Placeholder": itemgetter("Placeholder")
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    result = rag_chain.invoke({"Template":context,"Placeholder":Placeholder})
    return result


