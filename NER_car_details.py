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

# Path to the uploaded PDF file
pdf_file_path = r'C:\MyStuff\Descriptive_Brochure.pdf'

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
Question: Show me the complete information of Maserati GranTurismo
Answer: {example_answer_1}

Example 2:
Data: {example_data_2}
Question: Show me the technical specifications of Aston Martin Vantage
Answer: {example_answer_2}

Example 3:
Data: {example_data_3}
Question: Show me the interior features of Ferrari 488?
Answer: {example_answer_3}

Now, using only the information provided, answer the following question in the same format:
Data: {data}
Question: {question}
"""

# Provide example data in a concise format
example_data_1 = """
Maserati GranTurismo
The Maserati GranTurismo is the embodiment of Italian elegance and performance, designed for those who demand the very best.
Powered by a robust V6 4.5L gasoline engine, it delivers 680 horsepower and 433 Nm of torque, coupled with a manual transmission and FWD for an engaging driving experience.
Accelerating from 0 to 100 km/h in just 3.4 seconds and with a top speed of 352 km/h, the GranTurismo provides an exhilarating ride that is unmistakably Maserati.
Inside, Alcantara upholstery adds a luxurious touch to the cabin, while the Harman Kardon sound system immerses you in high-quality audio.
The interior is made from the finest sourced leather.
The infotainment system, equipped with a touchscreen and compatible with Apple CarPlay and Android Auto, keeps you connected wherever you go.
The GranTurismo is available in sophisticated colors including Red, Blue, Black, White, Silver, and Gray, allowing you to express your style.
Additional features such as heated seats, a sunroof, adaptive cruise control, and blind spot monitoring enhance both comfort and safety, making this brand-new
Maserati GranTurismo a compelling choice for those who appreciate refined performance and luxury.
The car is used with a milege of 81000 kms on it and is available for the price of $106K
"""
example_answer_1 = """
Model: Maserati Levante
Engine: V6, 4.5L
Drive Type: FWD
Fuel Type: Gasoline
Transmission: Manual
Power: 680 HP
Torque: 433 Nm
Top Speed: 352 km/h
Acceleration: 0-100 km/h in 3.4 seconds
Fuel Efficiency: 11 km/l
Color Options: Red, Blue, Black, White, Silver, Gray
Interior Material: Leather
Sound System: Harman Kardon
Infotainment System: Touchscreen, Apple CarPlay, Android Auto
Condition: Used
Mileage: 81000 km
Additional Features: Heated Seats, Sunroof, Adaptive Cruise Control, Blind Spot Monitoring
Price: $106k
"""

example_data_2 = """
Aston Martin Vantage
The Aston Martin Vantage is a masterpiece of British engineering, blending elegance with formidable power.
Equipped with a V8 3.0L gasoline engine, this Vantage generates 360 horsepower and 436 Nm of torque.
Its AWD system and automatic transmission deliver a thrilling driving experience, allowing it to reach a top speed of 320 km/h and go from 0 to 100 km/h in just 4.0 seconds.
Inside, the Vantage is adorned with luxurious leather seating and features a Bose Premium Sound system that fills the cabin with high-quality audio.
The touchscreen infotainment system, compatible with Apple CarPlay and Android Auto, ensures seamless connectivity on every drive. Available in striking colors—Red,
Blue, Black, White, Silver, and Gray—the Vantage combines refined style with unrivaled performance.
Additional features such as heated seats, a sunroof, adaptive cruise control, and blind spot monitoring add to the luxury and safety of this stunning Aston Martin Vantage.
It has 17000 kms milege on it and has the price set at $42k
"""
example_answer_2 = """
Model: Aston Martin Vantage
Engine: V8, 3.0L
Drive Type: AWD
Fuel Type: Gasoline
Transmission: Automatic
Power: 360 hP
Torque: 436 Nm
Top Speed: 320 km/h
Acceleration: 0-100 km/h in 4.0 seconds
"""

example_data_3 = """
Ferrari 488
The Ferrari 488 epitomizes Italian craftsmanship and thrilling performance, designed for those who crave the ultimate driving experience.
Under the hood, a powerful V6 3.5L gasoline engine produces 365 horsepower and 439 Nm of torque, perfectly matched with a manual transmission and FWD.
This high-performance sports car accelerates from 0 to 100 km/h in just 3.7 seconds, with a top speed of 226 km/h, delivering an exhilarating ride.
Inside, the Ferrari 488 offers premium Alcantara upholstery that exudes sophistication, paired with a Harman Kardon sound system for an immersive audio experience.
The touchscreen infotainment system, compatible with Apple CarPlay and Android Auto, provides seamless connectivity. Available in iconic Ferrari colors, including Red, Blue, Black, White, Silver, and Gray, the 488 is designed to make a statement.
Additional features such as heated seats, a sunroof, adaptive cruise control, and blind spot monitoring enhance both comfort and safety, making this brand-new Ferrari 488 an extraordinary blend of luxury and power.
The condition of the car is top notch and is brand new. It's price is set at $150k
Price: $43k
"""
example_answer_3 = """
Model: Ferrari 488
Color Options: Red, Blue, Black, White, Silver, Gray
Interior Material: Alcantara
Sound System: Harman Kardon
Infotainment System: Touchscreen, Apple CarPlay, Android Auto
Condition: Brand New
Mileage: 0 km
Additional Features: Heated Seats, Sunroof, Adaptive Cruise Control, Blind Spot Monitoring
"""

# Create the prompt template with input examples
prompt = PromptTemplate(
    input_variables=["example_data_1", "example_answer_1", "example_data_2", "example_answer_2", "example_data_3", "example_answer_3", "data", "question"],
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
    data=pdf_data,
    question="Show me the information of Porsche 911?"
)

# Call the LLM with the formatted prompt
response = ollama_llm.invoke(formatted_prompt)

# Print the response from the model
print(response)
