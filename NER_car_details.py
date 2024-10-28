import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def read_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def truncate_response(response_text, max_tokens=50):
    tokens = response_text.split()  # Tokenize by splitting words
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens]) + "..."
    return response_text

# Path to the uploaded PDF file
pdf_file_path = r'C:\MyStuff\Sample_dataset.pdf'

# Extract data from the PDF
pdf_data = read_pdf(pdf_file_path)

# Initialize the Ollama LLM
ollama_llm = Ollama(model="llama3.2")

# Define a few-shot prompt with examples and clear instructions
prompt_template = """
You are provided with factual car specifications. Answer strictly with only the car model and the requested details, without adding any additional information or commentary. 
If any requested detail is missing, reply with "Data not available."

Examples:

Example 1:
Data: {example_data_1}
Question: What is the Engine and Power of Maserati GranTurismo?
Answer:
Maserati GranTurismo
Engine: V6 4.5L
Power: 355 HP

Example 2:
Data: {example_data_2}
Question: What is the top speed and torque of Aston Martin Vantage?
Answer:
Aston Martin Vantage
Top Speed: 224 km/h
Torque: 436 Nm

Example 3:
Data: {example_data_3}
Question: What is the transmission and drive type of Ferrari 488?
Answer:
Ferrari 488
Transmission: Manual
Drive Type: FWD

Example 4:
Data: {example_data_4}
Question: What is the torque and top speed of an Audi Q7?
Answer:
Audi Q7
Torque: Data not available
Top Speed: 210 km/h

Now, using only the information provided, answer the following question in the same format:
Data: {data}
Question: {question}
"""

# Provide example data in a concise format
example_data_1 = """
Maserati GranTurismo
The Maserati GranTurismo is equipped with a V6 4.5L gasoline engine, producing 355 HP and 433 Nm of torque.
It has a manual transmission, FWD, and accelerates from 0 to 100 km/h in 3.4 seconds, with a top speed of 222 km/h.
Additional features include heated seats, sunroof, adaptive cruise control, and blind spot monitoring.
"""
example_answer_1 = """Maserati GranTurismo
Engine: V6 4.5L
Power: 355 HP"""

example_data_2 = """
Aston Martin Vantage
This model has a V8 3.0L gasoline engine with 360 HP and 436 Nm torque.
It features an AWD system, automatic transmission, top speed of 224 km/h, and accelerates from 0 to 100 km/h in 4.0 seconds.
It includes heated seats, sunroof, adaptive cruise control, and blind spot monitoring.
"""
example_answer_2 = """Aston Martin Vantage
Top Speed: 224 km/h
Torque: 436 Nm"""

example_data_3 = """
Ferrari 488
The Ferrari 488 has a V6 3.5L gasoline engine, 365 HP, 439 Nm torque, and manual transmission.
It’s equipped with FWD, accelerates from 0 to 100 km/h in 3.7 seconds, and reaches a top speed of 226 km/h.
"""
example_answer_3 = """Ferrari 488
Transmission: Manual
Drive Type: FWD"""

example_data_4 = """
Audi Q7
The Audi Q7 includes a V8 engine and automatic transmission, reaching a top speed of 210 km/h.
"""

example_answer_4 = """Audi Q7
Torque: Data not available
Top Speed: 210 km/h"""

# Create the prompt template with input examples
prompt = PromptTemplate(
    input_variables=["example_data_1", "example_answer_1", "example_data_2", "example_answer_2", "example_data_3", "example_answer_3", "example_data_4", "example_answer_4", "data", "question"],
    template=prompt_template
)

# Format the prompt with all the extracted PDF content
formatted_prompt = prompt.format(
    example_data_1=example_data_1,
    example_answer_1=example_answer_1,
    example_data_2=example_data_2,
    example_answer_2=example_answer_2,
    example_data_3=example_data_3,
    example_answer_3=example_answer_3,
    example_data_4=example_data_4,
    example_answer_4=example_answer_4,
    data=pdf_data,
    question="What is the top speed and the torque of Lexus LC500?"
)

# Call the LLM with the formatted prompt
response = ollama_llm.invoke(formatted_prompt)

truncated_response = truncate_response(response, max_tokens=50)

# Print the response from the model
print(truncated_response)
