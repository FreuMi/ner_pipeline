import ollama
import os
import re
import rdflib

def get_query_verify_unit(sentence, word):  
    query = f""" 
        The task is to verify whether the word is a correct unit of measure entity extracted from the given sentence.
        The input sentence: {sentence}
        Is the word "{word}" in the input sentence the correct unit entity? Please only answer with yes or no.
        
        Examples:
        sentence: "degree Celsius"
        word: "degree Celsius"
        output: yes
        
        sentence: "The realtive humidity is measured in %"
        word: "%"
        output: yes
        
        sentence: "The thermometer measures temperature"
        word: "power"
        output: no
        """
    return query

def get_query_verify_obs(sentence, word):  
    query = f""" 
        The task is to verify whether the word is a observed property entity extracted from the given sentence.
        The input sentence: {sentence}
        Is the word "{word}" in the input sentence the correct observable property entity? Please only answer with yes or no.
        
        Examples:
        sentence: "power"
        word: "power"
        output: yes
        
        sentence: "The thermometer measures temperature"
        word: "temperature"
        output: yes
        
        sentence: "The thermometer measures temperature"
        word: "power"
        output: no
        """
    return query

def get_query_unit(sentence):
    query = f""" 
        You are a linguist. The task is to label unit of measure and ratio % entities in the given sentence. 
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
        
    return query

def get_query_obs_property(sentence):
    query = f""" 
        You are a linguist. The task is to label observed property entities in the given sentence. 
        
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
        
    return query

def query(sentence):
    response = ollama.chat(model='gemma2:27b', messages=[
        {
            'role': 'user',
            'content': 'Do you know what Named Entity Recognition is?',
        },
        {
            'role': 'assistant',
            'content': 'Yes, Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, unit, etc.',
        },
        {
            'role': 'user',
            'content': 'Nice! Then i want you to pretend to be a linguist. The task is to label entities in the given sentence.',
        },
        {
            'role': 'assistant',
            'content': 'Got it, i do my best using NER!',
        },
        {   
            'role': 'user',
            'content': f'{sentence}',
        },
    
    ])
    
    res = response['message']['content']
    return res

# Define SPARQL query to get td:name and td:description
sparql_query = """
PREFIX td: <https://www.w3.org/2019/wot/td#>
SELECT DISTINCT ?name ?description
WHERE { 
  ?x td:name ?name .
  OPTIONAL { ?x td:description ?description . }
}
"""

def are_arrays_equal(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    return sorted(arr1) == sorted(arr2)

def read_lines_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
        return lines
    except Exception as e:
        print(e)
        return str(e)

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(e)
        return str(e)

def get_all_files_in_directory(directory):
    try:
        # Get list of all files and directories in the directory
        files_and_dirs = os.listdir(directory)
        
        # Filter out directories, keep only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        
        sorted_files = sorted(files)
        
        return sorted_files
    except Exception as e:
        return str(e)

def get_entities(res):
    # Extract the content inside @@ and ##
    tags = re.findall(r'@@(.*?)##', res)
    return tags

def true_or_false(res):
    if res == "Yes" or res == "yes":
        return True
    else:
        print(res)
        return False

def inference(sentence):
    final_result_units = []
    final_result_obs_prop = []
    
    ### UNIT ###
    unit_query = get_query_unit(sentence)

    res_unit = query(unit_query)
    res_unit = get_entities(res_unit)
    
    for element in res_unit:
        unit_validate_query = get_query_verify_unit(sentence, element)
        res_validate = query(unit_validate_query)
        eval = true_or_false(res_validate.strip())
        #print("Detected Unit:", element, "which is:", eval)
        if eval == True:
            final_result_units.append(element)
    
    # Obs. Property
    obs_prop_query = get_query_obs_property(sentence)
    
    res_prop = query(obs_prop_query)
    res_prop = get_entities(res_prop)
    
    for element in res_prop:
        prop_validate_query = get_query_verify_obs(sentence, element)
        res_validate = query(prop_validate_query)
        eval = true_or_false(res_validate.strip())
        #print("Detected Obs Property:", element, "which is:", eval)
        if eval == True:
            final_result_obs_prop.append(element)

    
    label = []
    for element in final_result_units:
        label.append("unit")
    
    for element in final_result_obs_prop:
        label.append("obs_prop")
    
    res = final_result_units + final_result_obs_prop
    
    return res, label
    
    
# Load file 
file_entries = read_file("./dataset/R1_dataset.txt").split("\n")

for sentence in file_entries:
        print("Working on", sentence)
        # Inference
        found_elements, found_elements_labels = inference(sentence)
        if len(found_elements) != 0:
            print(found_elements, found_elements_labels)
    