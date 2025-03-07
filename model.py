import requests
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

DATA_FILE = "qa_data.json"
MODEL_NAME = "t5-small"


def fetch_questions():
    url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?item ?itemLabel ?description
    WHERE {
      ?item wdt:P31 wd:Q2095;  
            rdfs:label ?itemLabel;
            schema:description ?description.
      FILTER(LANG(?itemLabel) = "ru")  
      FILTER(LANG(?description) = "ru")  
    }
    LIMIT 10000
    """

    params = {
        "query": query,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        questions = []
        existing_questions = set()

        for result in data["results"]["bindings"]:
            q_text = result["itemLabel"]["value"]
            q_description = result["description"]["value"] if "description" in result else "Нет описания"

            if q_text not in existing_questions:
                questions.append({"question": q_text, "answer": q_description})
                existing_questions.add(q_text)

        return questions

    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        time.sleep(10)
        return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


def train_on_data(data):
    model.train()
    for item in data:
        question = item["question"]
        answer = item.get("answer", "Нет ответа")
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        labels = tokenizer(answer, return_tensors="pt", padding=True, truncation=True).input_ids
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")


def generate_answer(question):
    model.eval()
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    while True:
        print("Процесс обучения начат...")
        questions = fetch_questions()
        existing_data = load_data()

        existing_question_texts = {item["question"] for item in existing_data}
        new_questions = [q for q in questions if q["question"] not in existing_question_texts]

        existing_data.extend(new_questions)
        save_data(existing_data)

        train_on_data(existing_data)
        print("Процесс обучения завершён! Запущен новый цикл...\n")
