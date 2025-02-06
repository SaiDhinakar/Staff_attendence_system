import requests


def store_embeddings(db_path, output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:8000/store_embeddings/"
    payload = {"db_path": db_path, "output_file": output_file}
    response = requests.post(url, json=payload)  # Sending as JSON body
    return response.json()


def load_embeddings(output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:8000/load_embeddings/"
    params = {"input_file": output_file}
    response = requests.get(url, params=params)
    return response.json()


if __name__ == "__main__":
    dataset_path = "dataset"
    embeddings_file = "face_embeddings.json"

    # Store embeddings
    store_response = store_embeddings(dataset_path, embeddings_file)
    print("Store Embeddings Response:", store_response)

    # Load embeddings
    load_response = load_embeddings(embeddings_file)
    print("Load Embeddings Response:", load_response)
