import os
import re

INPUT_PATH = "preprocessing/data/final_train_data.txt"
OUTPUT_Q = "preprocessing/data/questions.txt"
OUTPUT_A = "preprocessing/data/answers.txt"

# Regex to extract Q: ... and A: ...
QUESTION_RE = re.compile(r"Q:\s*(.*?)\s*A:", re.DOTALL)
ANSWER_RE = re.compile(r"A:\s*(.*?)\s*(\[EOS\]|$)", re.DOTALL)

def extract_question(line):
    m = QUESTION_RE.search(line)
    return m.group(1).strip() if m else None

def extract_answer(line):
    m = ANSWER_RE.search(line)
    return m.group(1).strip() if m else None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"File not found: {INPUT_PATH}")
        return
    
    questions = []
    answers = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            q = extract_question(line)
            a = extract_answer(line)

            if q is None or a is None:
                print("Skipping malformed line:", line[:80], "...")
                continue

            questions.append(q)
            answers.append(a)

    print(f"Extracted {len(questions)} question–answer pairs.")

    # Write outputs
    with open(OUTPUT_Q, "w", encoding="utf-8") as fq:
        for q in questions:
            fq.write(q + "\n")

    with open(OUTPUT_A, "w", encoding="utf-8") as fa:
        for a in answers:
            fa.write(a + "\n")

    print(f"Saved:")
    print(f"  - {OUTPUT_Q}")
    print(f"  - {OUTPUT_A}")
    print("Done.")

if __name__ == "__main__":
    main()