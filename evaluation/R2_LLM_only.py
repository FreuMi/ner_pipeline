import ollama
from sentence_transformers import SentenceTransformer
import re
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
    
def self_verification_unit(sentence, word):
    
    query = f""" 
        The task is to tell me the unit of measure with the symbol "{word}" in the context of the input sentence.
        The input sentence: {sentence}
        
        Allowed units are:
        "Siemens", "Second", "Ampere", "Angstrom", "Gram", "Tesla", "Voltage", "cubic meter", "liter", "meter", "ton", "kelvin", "acceleration g", "nanometer", "hertz", "kilo Pascal", "meter per second", "electronvolt", "degree Celsisus", "decibel", "kilo hertz".
        
        Please only answer with the name of the unit. If the unit makes no sense in the context answer "none".
        
        Example 1:
        Sentence: mass m is measured in g.
        Symbol: g
        Answer: Gram
        
        Example 2:
        Sentence: mass m is measured in g.
        Symbol: m
        Answer: none
        """
    response = ollama.chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': f'{query}',
        },
    ])
    res = response['message']['content']
    print("Verification result:", res)
    
    return res

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

qudt_unit_uris = {
        "siemens": "https://qudt.org/vocab/unit/S",
        "second": "https://qudt.org/vocab/unit/SEC",
        "ampere": "https://qudt.org/vocab/unit/A",
        "angstrom": "https://qudt.org/vocab/unit/ANGSTROM",
        "gram": "https://qudt.org/vocab/unit/GM",
        "tesla": "https://qudt.org/vocab/unit/T",
        "voltage": "https://qudt.org/vocab/unit/V",
        "cubic meter": "https://qudt.org/vocab/unit/M3",
        "liter": "https://qudt.org/vocab/unit/L",
        "meter": "https://qudt.org/vocab/unit/M",
        "ton": "https://qudt.org/vocab/unit/TON_Metric",
        "kelvin": "https://qudt.org/vocab/unit/K",
        "acceleration g": "https://qudt.org/vocab/unit/G",
        "nanometer": "https://qudt.org/vocab/unit/NanoM",
        "hertz": "https://qudt.org/vocab/unit/HZ",
        "kilo pascal": "https://qudt.org/vocab/unit/KiloPA",
        "meter per second": "https://qudt.org/vocab/unit/M-PER-SEC",
        "electronvolt": "https://qudt.org/vocab/unit/EV",
        "degree celsius": "https://qudt.org/vocab/unit/DEG_C",
        "decibel": "https://qudt.org/vocab/unit/DeciB",
        "kilo hertz": "https://qudt.org/vocab/unit/KiloHZ",
        "none": ""
    }

if __name__ == "__main__":
    # Load dataset  
    file_entries = read_file("./dataset/R2_dataset.txt").split("\n")

    start_total_time = time.time()
    with open("./results_R2_llm_only.txt", "a") as rf:
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
            final_units = []
            
            for res in res_arr:
                if res == "":
                    continue
            
                result = self_verification_unit(sentence, res)
                
                final_units.append(result)
                
            stringResult = ""
            for element in final_units:
                try:
                    stringResult = stringResult + qudt_unit_uris[element.strip().lower()] + ";"
                except:
                    stringResult = stringResult + element.strip() + ";"
            stringResult += "\n"   
            rf.write(stringResult)     
            rf.flush() 
            end_time = time.time()
            print("Took ", end_time-start_time)
            
    stop_total_time = time.time()
    
    print("Total Time needed:", stop_total_time-start_total_time)
    print("On average per sentence:", (stop_total_time-start_total_time)/cnt)