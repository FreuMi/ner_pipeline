import faiss
import numpy as np

def create_db(model):
    # QUDT units and URIs, including abbreviations
    qudt_units = ["Siemens", "S", "Second", "s", "Ampere", "A", "Angstrom", "Å", "Gram", "gm", \
                  "Tesla", "T", "Voltage", "V", "m^3", "l", "m", "t", "K", "g", "nm"]
    
    qudt_unit_uris = {
        "Siemens": "https://qudt.org/vocab/unit/S",
        "S": "https://qudt.org/vocab/unit/S",
        "Second": "https://qudt.org/vocab/unit/SEC",
        "s": "https://qudt.org/vocab/unit/SEC",
        "Ampere": "https://qudt.org/vocab/unit/A",
        "A": "https://qudt.org/vocab/unit/A",
        "Angstrom": "https://qudt.org/vocab/unit/ANGSTROM",
        "Å": "https://qudt.org/vocab/unit/ANGSTROM",
        "Gram": "https://qudt.org/vocab/unit/GM",
        "gm": "https://qudt.org/vocab/unit/GM",
        "Tesla": "https://qudt.org/vocab/unit/T",
        "T": "https://qudt.org/vocab/unit/T",
        "Voltage": "https://qudt.org/vocab/unit/V",
        "V": "https://qudt.org/vocab/unit/V",
        "m^3": "https://qudt.org/vocab/unit/M3",
        "l": "https://qudt.org/vocab/unit/L",
        "m": "https://qudt.org/vocab/unit/M",
        "t": "https://qudt.org/vocab/unit/TON_Metric",
        "K": "https://qudt.org/vocab/unit/K",
        "g": "https://qudt.org/vocab/unit/G",
        "nm": "https://qudt.org/vocab/unit/NanoM"
    }

    # observable properties and Wikidata URIs
    observable_properties = ["temperature", "pressure", "humidity"]
    observable_property_uris = {
        "temperature": "https://www.wikidata.org/wiki/Q11466",
        "pressure": "https://www.wikidata.org/wiki/Q11574",
        "humidity": "https://www.wikidata.org/wiki/Q37876"
    }

    # Combnine
    #all_terms = qudt_units + observable_properties
    #all_term_uris = {**qudt_unit_uris, **observable_property_uris}
    all_terms = qudt_units
    all_term_uris = {**qudt_unit_uris}
    all_embeddings = {term: model.encode(term) for term in all_terms}
    all_embeddings_array = np.array([emb for emb in all_embeddings.values()]).astype('float32')

    # Create a Faiss index with L2 distance for similarity search
    dimension = all_embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension) 

    # Add all embeddings to index
    index.add(all_embeddings_array)

    # Map index positions to URIs
    all_term_map = {i: uri for i, uri in enumerate(all_term_uris.values())}
    
    return index, all_term_map

# Function to find the closest QUDT units
def find_closest_qudt_units(word, model, index, qudt_unit_map, top_k=3):
    embedding = model.encode(word).reshape(1, -1).astype('float32')
    distances, indices = index.search(embedding, top_k)
    return [(qudt_unit_map[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
