import sys

true_positives = 0
false_positives = 0
false_negatives = 0

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error: {e}")
        return str(e) 

# Load results
generated_file_data = read_file("./results_R2_pipeline.txt").split("\n")
expected_file_data = read_file("./dataset/R2_dataset_exp.txt").split("\n")

# Check if length is the same
if len(generated_file_data) != len(expected_file_data):
    print("Data size missmatch!")
    sys.exit()

for i in range(len(generated_file_data)):
    generated = generated_file_data[i]
    expected = expected_file_data[i]
    
    generated_split = generated.split(";")
    expected_split = expected.split(";")
    
    # Get true positives
    for element in generated_split:
        if element == "":
            continue
        
        if element in expected_split:
            true_positives += 1
            
    # Get false positives
    for element in generated_split:
        if element == "":
            continue
        
        if element not in expected_split:
            false_positives += 1
            
    # Get false negatives
    for element in expected_split:
        if element == "":
            continue
        
        if element not in generated_split:
            false_negatives += 1
    
    print("TP", true_positives)
    print("FP", false_positives)
    print("FN", false_negatives)
    
    
    print("Generated:", generated_split)
    print("Expected:", expected_split)
    
    
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

f1 = 2* ((precision * recall) / (precision + recall))

print("Precision",precision)
print("Recall",recall)
print("f1",f1)

    