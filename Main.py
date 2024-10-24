import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

dataset = pd.read_csv("C:\MyStuff\CAR_DETAILS.csv")
ollama_llm = Ollama(model = "phi")

prompt_template = """
Use the following data to answer the question:
Data: {data}
Question: {question}
"""

prompt = PromptTemplate(input_variables=["data","question"], template=prompt_template)

result = prompt | ollama_llm
response = result.invoke({"data": dataset.head(10).to_string(),
                          "question": "What is the most km_driven car?"})

print(response)