from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import re
import vec_db
import requests

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

def self_verification_unit(sentence, word, url):
    # Get RDF definition
    r = requests.get(url)
    info = r.text
    
    query = f""" 
        The task is to verify whether the word is the correct unit entity extracted from the given sentence.
        The input sentence: {sentence}
        Is the word "{word}" in the input sentence the correct unit entity? Please only answer with yes or no.
        
        Here is the definition of "{word}": {info}
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

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create in memory vector db
index, qudt_unit_map = vec_db.create_db(model)

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

        
        relevant_entires = []
        top_k = 3
        closest_qudt_units = vec_db.find_closest_qudt_units(res, model, index, qudt_unit_map, top_k)
        print(f"The closest QUDT unit(s) to '{res}':")
        for uri, distance in closest_qudt_units:
            #print(f"QUDT URI: {uri}, Distance: {distance}")
            if distance < 0.85:
                print(f"QUDT URI: {uri}, Distance: {distance}")
                relevant_entires.append(uri)
                
        entries[res] = relevant_entires
     
    # Disambiguate and Validate
    for ent in entries:
        if len(entries[ent]) == 0:
            continue

        for url in entries[ent]:
            self_verification_unit(sentence, ent, url)
    