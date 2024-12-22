import ollama
import os

model = "llama3.2"

# path to input and output files:
input_file = "/home/krishna/Documents/Ollama/data/grocery_list.txt"
output_file = "/home/krishna/Documents/Ollama/data/categorized_grocery_list.txt"

# check if input file exists:
if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found")
    exit(1)

# read the uncategorized grocery items from the input file:
with open(input_file, "r") as f:
    items = f.read().strip()

# prompt:
prompt = f"""
    you are an assistant that categorizes and sorts grocery items.
    
    Here is a list of grocery items:

    {items}

    Please:

    1. Categorize these items into appropiate categories such as produce, dairy, meat, bakery, beverages, etc.
    2. Sort the items alphabetically within each category.
    3. Present the categorized list in a clear and organized manner, using bullet points and numbering.

    """

    # Send the prompt and get the response:
try:
    response = ollama.generate(model = model, prompt = prompt)
    generated_text = response.get("response", "")
    print("Categorized List: ")
    print()
    print(generated_text)

    # Write the categorized list to the output file:
    with open(output_file, "w") as f:
    	f.write(generated_text.strip())

    print(f"Categorized grocery list has been saved to: '{output_file}'.")
except Exception as e:
	print("An error occured: ", str(e))
