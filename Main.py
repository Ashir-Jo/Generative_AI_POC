import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

#Path to the csv file
dataset = pd.read_csv("C:\MyStuff\CAR_DETAILS.csv")

#Initializing ollama LLM
ollama_llm = Ollama(model = "phi")

# One shot prompt template
prompt_template = """
Use the following data to answer the question:
Data: {data}
Question: {question}
"""

#prompt template with the input variables
prompt = PromptTemplate(input_variables=["data","question"], template=prompt_template)

#Call Ollama LLM
result = prompt | ollama_llm
response = result.invoke({"data": dataset.head(10).to_string(),
                          "question": "What is the most km_driven car?"})

#print the output
print(response)
