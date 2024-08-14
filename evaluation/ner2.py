import spacy
import re

nlp = spacy.load("./output/model-best")

def clear_number(item):
    # Define pattern for numbers, whitespace, +, and - at the beginning and end
    pattern = r'^[0-9\s\[\]\(\)\{\}\+\-]+|[0-9\s\[\]\(\)\{\}\+\-]+$'
    return re.sub(pattern, '', item)

def clear_brackets(input_string):
    brackets = "[](){}"
    for bracket in brackets:
        input_string = input_string.replace(bracket, "")
    return input_string

def inference(sentence):
    sentence = sentence.lower()
    #print("Sentence:", sentence)
    doc = nlp(sentence)
    found_elements = []
    found_elements_labels = []
    for ent in doc.ents:
        res_ent = clear_number(str(ent))
        if res_ent != "":
            #print("Detected:", res_ent, ent.label_)
            found_elements.append(res_ent)
            found_elements_labels.append(ent.label_)
    
    return found_elements, found_elements_labels
         