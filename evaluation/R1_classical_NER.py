import ner2
import time

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error: {e}")
        return str(e) 
    
    
file_entries = read_file("./dataset/R1_dataset.txt").split("\n")
start_total_time = time.time()
for sentence in file_entries:
    print("========")
    print("Working on", sentence)
     
    print(ner2.inference(sentence))
    
stop_total_time = time.time()
    
print("Total Time needed:", stop_total_time-start_total_time)
print("On average per sentence:", (stop_total_time-start_total_time)/len(file_entries))