import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def read_pdf(file_path):
    # Function to extract text from the provided PDF file
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Path to the uploaded PDF file
pdf_file_path = 'C:\MyStuff\detailed_car_specifications.pdf'

# Extract data from the PDF
pdf_data = read_pdf(pdf_file_path)

# Initialize the Ollama LLM
ollama_llm = Ollama(model="phi")

# Define a few-shot prompt with multiple examples
prompt_template = """
Here are some examples of how to extract car models based on specifications:

Example 1:
Data: {example_data_1}
Question: Show me all the models which have hybrid engines.
Answer: {example_answer_1}

Example 2:
Data: {example_data_2}
Question: Which models come with all-wheel drive?
Answer: {example_answer_2}

Example 3:
Data: {example_data_3}
Question: Which models have horsepower greater than 300?
Answer: {example_answer_3}

Now, use the following data to answer the question:
Data: {data}
Question: {question}
"""

# Provide example data for few-shot learning
example_data_1 = """
Model: Ford Mustang
Engine: V8, 3.0L
Drive Type: AWD
Fuel Type: Hybrid
Transmission: Automatic
Power: 300 HP
Torque: 400 Nm
Top Speed: 200 km/h
Acceleration: 0-100 km/h in 4.0 seconds
Fuel Efficiency: 10 km/l
Color Options: Red, Blue, Black, White, Silver, Gray
Interior Material: Leather
Sound System: Bose Premium Sound
Infotainment System: Touchscreen, Apple CarPlay, Android Auto
Condition: Used, Mileage: 5000 km
Price: $30k
"""
example_answer_1 = "Ford Mustang"

example_data_2 = """
Model: Dodge Charger
Engine: V8, 4.0L
Drive Type: AWD
Fuel Type: Gasoline
Transmission: Automatic
Power: 310 HP
Torque: 406 Nm
Top Speed: 204 km/h
Acceleration: 0-100 km/h in 3.4 seconds
Fuel Efficiency: 12 km/l
Color Options: Red, Blue, Black, White, Silver, Gray
Interior Material: Leather
Sound System: Harman Kardon
Infotainment System: Touchscreen, Apple CarPlay, Android Auto
Condition: Brand New, Mileage: 0 km
Price: $32k
"""
example_answer_2 = "Dodge Charger"

example_data_3 = """
Model: Chevrolet Camaro
Engine: V6, 3.5L
Drive Type: FWD
Fuel Type: Gasoline
Transmission: Manual
Power: 305 HP
Torque: 403 Nm
Top Speed: 202 km/h
Acceleration: 0-100 km/h in 3.7 seconds
Fuel Efficiency: 11 km/l
Color Options: Red, Blue, Black, White, Silver, Gray
Interior Material: Alcantara
Sound System: Harman Kardon
Infotainment System: Touchscreen, Apple CarPlay, Android Auto
Condition: Brand New, Mileage: 0 km
Price: $31k
"""
example_answer_3 = "Chevrolet Camaro"

# Create the prompt template with input examples
prompt = PromptTemplate(
                        
    #input_variables=["example_data_1", "example_answer_1", "example_data_2", "example_answer_2", "example_data_3", "example_answer_3", "data", "question"],
    input_variables=["data", "question"],
    template=prompt_template
)

# Format the prompt with all the extracted PDF content
formatted_prompt = prompt.format(
    #example_data_1=example_data_1,
    #example_answer_1=example_answer_1,
    #example_data_2=example_data_2,
    #example_answer_2=example_answer_2,
    #example_data_3=example_data_3,
    #example_answer_3=example_answer_3,
    data=pdf_data,  # Use the entire extracted PDF data
    question="Show me all the models which has V8 engines"
)

# Call the LLM with the formatted prompt
response = ollama_llm.invoke(formatted_prompt)

# Print the response from the model
print(response)
