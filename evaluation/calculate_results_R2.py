import sys

RESULT_FILE_PATH = "./results_R2_pipeline.txt"
RESULT_REFERENCE_FILE_PATH = "./dataset/R2_dataset_exp.txt"

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
generated_file_data = read_file(RESULT_FILE_PATH).split("\n")
expected_file_data = read_file(RESULT_REFERENCE_FILE_PATH).split("\n")

# Check if length is the same
if len(generated_file_data) != len(expected_file_data):
    print("Data size missmatch!")
    sys.exit()

cnt = 0
for i in range(len(generated_file_data)):
    generated = generated_file_data[i]
    expected = expected_file_data[i]
    
    generated_split = generated.split(";")
    expected_split = expected.split(";")
    
    cnt+=1
    print("Working on element", cnt)
    print("generated_split",generated_split)
    print("expected_split",expected_split)    
    
    # Get true positives
    expected_split_tmp = expected_split.copy()
    for element in generated_split:
        if element == "":
            continue
        
        if element in expected_split_tmp:
            true_positives += 1
            expected_split_tmp.remove(element)
            
    # Get false positives
    expected_split_tmp = expected_split.copy()
    for element in generated_split:
        if element == "":
            continue
        
        if element in expected_split_tmp:
            expected_split_tmp.remove(element) 
        else:
            false_positives += 1
            
    # Get false negatives
    generated_split_tmp = generated_split.copy()
    for element in expected_split:
        if element == "":
            continue
        
        if element in generated_split_tmp:
            generated_split_tmp.remove(element)
        else:
            false_negatives += 1
    
    print("TP", true_positives)
    print("FP", false_positives)
    print("FN", false_negatives)
    print("========")
    
    
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

f1 = 2* ((precision * recall) / (precision + recall))

print("Precision",precision)
print("Recall",recall)
print("f1",f1)