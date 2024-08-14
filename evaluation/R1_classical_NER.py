import ner2

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error: {e}")
        return str(e) 
    
    
file_entries = read_file("./dataset/R1_dataset.txt").split("\n")

for sentence in file_entries:
    print("========")
    print("Working on", sentence)
     
    print(ner2.inference(sentence))