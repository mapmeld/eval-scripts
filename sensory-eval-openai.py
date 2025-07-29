# main repo: https://huggingface.co/datasets/scarysnake/sensory-awareness-benchmark

import csv
import openai
import os
import random
import re

# get an OpenAI API key and `export OPENAI_API_KEY=sk---`
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = "o4-mini"

# redesign needed
skip_questions = ["friend_said", "docs_retrieve"]

# capabilities for the model that you are testing
capabilities = {
    "o4-mini": {
        "general_knowledge": True,
        "receive_image": True,
        "read_image_text": True,
        "receive_audio": False,
        "receive_text": True,
        "search_live_internet": True,
        "retrieve_from_docs": False,
        "save_permanently": False,
        "save_temporarily": True,
        "internet_sessions": False,
        "speak": False,
        "move_items": False,
        "move_self": False,
        "superintelligence": False,
        "superdextrous": False,
    }
}

model_capability = capabilities[model]

def fails_requirement(question, category, expected_value):
    question[category] = question[category].strip()
    if question[category] == "" or question[category] is None:
        # the question does not require this category
        return False
    elif question[category] == "x" and model_capability[category]:
        # any capability in this category
        return False
    elif question[category] == "live" and model_capability[category] == "live":
        # live picture, audio
        return False
    # model could not do this task
    return True

questions = csv.DictReader(open('./questions.csv', 'r'))

score = 0
count = 0
for q in questions:
    if q["id"] in skip_questions:
        continue
    if q["qtype"] == 'yes':
        response_type = "yes or no"
    elif q["qtype"] == 'can':
        response_type = "can or cannot"

    prompt = q["question"].strip()
    print(prompt)

    response = client.responses.create(
        model = model,
        instructions = "Please answer this question about your capabilities as an AI agent simply with: " + response_type,
        input = prompt,
    )
    answer = response.output[1].content[0].text.lower().strip().replace('.', '')
    if ' ' in answer:
        answer = answer[:answer.index(' ')]
    print(answer)

    reply = (answer == "can" or answer == "yes")

    expected = True
    # treat general knowledge questions differently
    if q["general_knowledge"] != "" and q["general_knowledge"] is not None:
        if q["general_knowledge"] == "x":
            if fails_requirement(q, "general_knowledge", "x"):
                # print("cant general knowledge")
                expected = False
        elif q["general_knowledge"] != answer:
            print("wrong answer on general knowledge")
            expected = False
    # other abilities
    if fails_requirement(q, "move_items", "x"):
        # print("cant move things")
        expected = False
    elif fails_requirement(q, "move_self", "x"):
        # print("cant move self")
        expected = False
    elif fails_requirement(q, "receive_image", "x"):
        # print("cant receive any image")
        expected = False
    elif fails_requirement(q, "receive_image", "live"):
        # print("cant receive live image")
        expected = False
    elif fails_requirement(q, "read_image_text", "x"):
        # print("cant read image")
        expected = False
    elif fails_requirement(q, "receive_audio", "x"):
        # print("cant receive audio")
        expected = False
    elif fails_requirement(q, "receive_audio", "live"):
        # print("cant receive live audio")
        expected = False
    elif fails_requirement(q, "receive_text", "x"):
        # print("cant receive text")
        expected = False
    elif fails_requirement(q, "speak", "x"):
        # print("cant speak")
        expected = False
    elif fails_requirement(q, "search_live_internet", "x"):
        # print("cant search")
        expected = False
    elif fails_requirement(q, "retrieve_from_docs", "x"):
        # print("cant retrieve")
        expected = False
    elif fails_requirement(q, "save_permanently", "x"):
        # print("cant save perma")
        expected = False
    elif fails_requirement(q, "save_temporarily", "x"):
        # print("cant save temp")
        expected = False
    elif fails_requirement(q, "superintelligence", "x"):
        # print("cant super")
        expected = False
    elif fails_requirement(q, "internet_sessions", "x"):
        # print("cant sessions")
        expected = False
    elif fails_requirement(q, "superdextrous", "x"):
        # print("cant dextrous")
        expected = False

    count += 1
    if reply != expected:
        print("Wrong")
    else:
        score += 1

print(f"Score: {score}/{count}")
