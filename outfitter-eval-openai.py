# main repo: https://huggingface.co/datasets/scarysnake/outfitter-advice

import csv
import openai
import os
import random
import re

# consistent use
random.seed(1929)

# get an OpenAI API key and `export OPENAI_API_KEY=sk---`
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = "o4-mini"

# dynamic list length returning "1, 2, or 3"; "1 or 2"
def natural_enum(n: int) -> str:
    nums = [str(i) for i in range(1, n + 1)]
    if n == 1:
        return nums[0]
    return ", ".join(nums[:-1]) + f", or {nums[-1]}"

def shuffle_with_index_tracking(images, best_index=0, second_index=1):
    copy = [i for i in images]
    random.shuffle(copy)
    return copy, copy.index(images[0]), copy.index(images[1])

def ask_openai_best_outfit(post, images):
    image_messages = [
        {
            "type": "input_image",
            "image_url": url
        } for url in images
    ]

    prompt = (
        "You're a fashion-savvy assistant. Here are images of different outfits. "
        f"{post['title']} {post['selftext']} "
        f"Respond only with the number of the best outfit: {natural_enum(len(images))}."
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    *image_messages
                ]}
            ],
        )
    except openai.BadRequestError as e:
        raise Exception("There is an issue possibly retrieving images for the post. It might have been deleted?")


    # for this model, output[0] is any reasoning
    text = response.output[1].content[0].text
    match = re.search(r"\d+", text)
    if match is not None:
        return int(match[0])
    raise ValueError(f"Invalid response: {text}")

posts = csv.DictReader(open('./dataset.csv', 'r'))
score = 0.0
items = 0
for post in posts:
    items += 1
    print(post['title'])
    shuffled_images, best_after_shuffle, second_after_shuffle = shuffle_with_index_tracking(post["images"].split(","))
    # print(f"{len(shuffled_images)} options")

    # this subtracts 1 so AI returning "1" is the 0th image
    predicted_index = ask_openai_best_outfit(post, shuffled_images) - 1

    if predicted_index == best_after_shuffle:
        score += 1.0
    elif predicted_index == second_after_shuffle:
        score += float(post["secondChoice"]) / float(post["firstChoiceVotes"])
    print(f"Total score: {score}/{items}")
