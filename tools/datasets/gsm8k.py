import datasets
import re
import os

# Two linebreaks between each question-answer pair, and one linebreak between the question and the answer
LM_EVAL_PROMPT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n
"""

SAVE_DIR = "./../../outputs/datasets"

def download_split(split_name):
    print(f"Initalizing GSM8K dataset ({split_name})...")
    dataset = datasets.load_dataset("gsm8k", "main", split=split_name)

    # Check if the file already exists
    if os.path.exists(f"{SAVE_DIR}/gsm8k_lm_eval_{split_name}.txt"):
        print(f"File already exists ({split_name}). Skipping...")
        return

    print("Writing LM eval prompts to file...")
    with open(f"{SAVE_DIR}/gsm8k_lm_eval_{split_name}.txt", "a") as file:
        print("Writing LM eval prompts to file...")
        for i, example in enumerate(dataset):
            question = example["question"].replace("\n", " ")
            answer = example["answer"].replace("\n", " ")

            # remove expressions of the form <<48/2=24>> from the answer
            answer = re.sub(r"<<.*?>>", "", answer)
            
            answer = answer.split("####")[0] + " The answer is " + answer.split("#### ")[1] + "."
            qa_text = "Q: " + question + "\nA: " + answer + "\n\n"

            file.write(qa_text)

if __name__ == "__main__":
    download_split("train")
    download_split("test")
    
    print("Done!")
        


