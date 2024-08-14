from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import re

def query_unit(sentence):
    query = f""" 
        I am an excellent linguist. The task is to label unit and ratio % entities in the given sentence. 
        The found entities should be enclosed in @@ENTITY## for easier parsing.
        Return only the result sentence.

        Below are some examples. Never output these, only the sentence with annotated entities.
        Input:The unit is degrees Celsius
        Output:The unit is @@degrees Celsius##
        Input:The property % is measured.
        Output:The property  @@%## is measured.
        Input:The area returns 5 m^3.
        Output:The area returns 5 @@m^3##.
        Input:Rare Hendrix song sells for $17
        Output:
        
        Sentence: "{sentence}" """

    res = llm.invoke(query)
    print("RES:", res)
    tags = get_entities(res)
    tags = clear_number(tags)
    return tags


def query_obs_property(sentence):
    query = f""" 
        I am an excellent linguist. The task is to label observed property entities in the given sentence. 
        
        The found entities should be enclosed in @@ENTITY## for easier parsing.
        Return only the result sentence.
        Below are some examples. Never output these, only the sentence with annotated entities.
        
        Input:The temperature is 20°C
        Output:The @@temperature## is 20°C
        Input:The action runs a long distance
        Output:The action runs a long @@distance##
        Input:Rare Hendrix song sells for $17
        Output:
        
        Sentence: "{sentence}" """

    res = llm.invoke(query)
    print("RES:", res)
    tags = get_entities(res)
    return tags
    
    
def self_verification_unit(sentence, word):
    # Get RDF definition
    #r = requests.get(url)
    #info = r.text   
    
    query = f""" 
        The task is to tell me the unit of measure with the symbol "{word}" in the context of the input sentence.
        
        The input sentence: {sentence}
        Please only answer with the name of the unit. If the unit makes no sense in the context answer None.
        
        Allowed units are:
        "Siemens", "Second", "Ampere", "Angstrom", "Gram", "Tesla", "Voltage", "cubic meter", "liter", "meter", "ton", "Kelvin", "acceleration g", "nm"
        """
    res = llm.invoke(query)
    print("Verification result:", res)

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error: {e}")
        return str(e) 

def clear_number(item):
    # Define patterns for numbers, whitespace, +, and - at the beginning and end
    pattern = r'^[0-9\s\[\]\(\)\{\}\+\-]+|[0-9\s\[\]\(\)\{\}\+\-]+$'
    if str(type(item)) == "<class 'list'>":
        new_list = []
        for element in item:
            new_list.append(re.sub(pattern, '', element))
        return new_list
    
    return re.sub(pattern, '', item)

def get_entities(res):
    # Extract the content inside @@ and ##
    tags = re.findall(r'@@(.*?)##', res)
    return tags

# Load llm model
llm = Ollama(model="gemma2:27b")

# Load file 
file_entries = read_file("./dataset/R2_dataset.txt").split("\n")

for sentence in file_entries:
    print("==================================")
    print("Working on:", sentence)
    res_arr = query_unit(sentence)
    print("RES:", res_arr)

    # Store results
    entries = {}
    
    # Find the closest QUDT units
    for res in res_arr:
        if res == "":
            continue
    
        self_verification_unit(sentence, res)