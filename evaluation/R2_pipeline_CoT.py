import ollama
from sentence_transformers import SentenceTransformer
import re
import vec_db
import requests
import sys
import time

MODEL = "gemma2:27b"

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

    response = ollama.chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': f'{query}',
        },
    ])
    
    res = response['message']['content']
    print("NER result:", res)
    tags = get_entities(res)
    tags = clear_number(tags)
    return tags

def self_verification_unit(sentence, symbol_in_txt, symbol_of_unit, url):
    # Get RDF definition
    info = send_request(url)
    print("Request finished.")
        
    query = f""" 
        Task: Determine whether the given knowledge graph unit entity "{symbol_of_unit}" has the same meaning and relevance as the detected unit entity "{symbol_in_txt}" in the context of the provided sentence. You must decide whether the two entities should be linked. Consider both the meaning of the unit and the context in which it appears.

        - Pay attention to prefixes and the context in which the unit is used.

        Input sentence: "{sentence}"
        Knowledge graph entity: "{symbol_of_unit}"
        Detected entity in sentence: "{symbol_in_txt}"

        Definition of "{symbol_of_unit}" in the knowledge graph: {info}

        Instructions:
        - Compare the detected entity "{symbol_in_txt}" in the sentence with the knowledge graph entity "{symbol_of_unit}".
        - Contextual Matching: Consider the context of the sentence to determine the meaning of "{symbol_in_txt}" in the sentence. Does it represent the same concept as the knowledge graph entity "{symbol_of_unit}"?
        - Definition Confirmation: Evaluate whether the definition of "{symbol_of_unit}" in the knowledge graph matches the meaning of "{symbol_in_txt}" in the sentence.
        - Relevance: Determine if the two entities should be linked based on whether they represent the same unit in the given context.

        Note: If the meaning of "{symbol_in_txt}" in the sentence does not align with the definition provided by the knowledge graph, the entities should not be linked.

        Example 1:
        - Sentence: "mass m is measured in g."
        - Knowledge graph entity: "GM"
        - Detected entity: "g"
        - Definition: "A unit of mass in the metric system."
        - Answer: The detected entity "g" refers to grams, which matches the knowledge graph entity "GM." Therefore, the entities should be linked.

        Example 2:
        - Sentence: "f in Hz."
        - Knowledge graph entity: "kiloHZ"
        - Detected entity: "Hz"
        - Definition: "Kilohertz (kHz) is a unit of frequency."
        - Answer: The detected entity "Hz" refers to hertz, not kilohertz. The entities should not be linked.
 """    
    response = ollama.chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': f'{query}',
        },
    ])
    res_tmp = response['message']['content']
    print("TEMP for", url, res_tmp)
    messages=[
        {
            'role': 'user',
            'content': f'{query}',
        },
        {
            'role': 'assistant',
            'content': f'{res_tmp}',
        },
        {
            'role': 'user',
            'content': 'So please give me the final decision if the entity should be linked. Only answer "yes" or "no"',
        },
    ]
    
    response = ollama.chat(model=MODEL, messages=messages)
    
    res = response['message']['content']    
        
    print("Verification result:", res)
    
    if res.strip() == "yes" or res.strip() == "Yes":
        print("ret: True")
        return True
    else:
        print("ret: False")
        return False
    
def send_request(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        info = r.text
        return info 
    except requests.exceptions.Timeout:
        print("url:",url)
        print("The request timed out")
        sys.exit()
    except requests.exceptions.HTTPError as http_err:
        print("url:",url)
        print(f"HTTP error occurred: {http_err}")
        sys.exit()
    except requests.exceptions.RequestException as err:
        print("url:",url)
        print(f"Other error occurred: {err}")
        sys.exit()

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error: {e}")
        return str(e) 

def clear_number(item):
    # Define a pattern to match digits, whitespace, ., +, -, and brackets at any position in the string
    pattern = r'[0-9\s\[\]\(\)\{\}\+\-\.]+'
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

if __name__ == "__main__":
    # Load pre trained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create in memory vector db for similarity search
    index, qudt_unit_map = vec_db.create_db(model)

    # Load dataset 
    file_entries = read_file("./dataset/R2_dataset.txt").split("\n")
    
    start_total_time = time.time()

    with open("./results_R2_pipeline_CoT.txt", "a") as rf:
        cnt = 0
        for sentence in file_entries:
            start_time = time.time()
            print("==================================")
            cnt += 1
            print("No.", cnt)
            print("Working on:", sentence)
            print("==================================")
            res_arr = query_unit(sentence)
            print("Detected units:", res_arr)

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
            final_units = []
            for ent in entries:
                if len(entries[ent]) == 0:
                    continue

                for url in entries[ent]:
                    res = self_verification_unit(sentence, ent, url.split("/")[-1], url)
                    if res == False:
                        continue
                    final_units.append(url)
                    
            stringResult = ""
            for element in final_units:
                stringResult = stringResult + element + ";"
            stringResult += "\n"   
            rf.write(stringResult)     
            rf.flush() 
            end_time = time.time()
            print("Took ", end_time-start_time)

    stop_total_time = time.time()
    
    print("Total Time needed:", stop_total_time-start_total_time)
    print("On average per sentence:", (stop_total_time-start_total_time)/cnt)