
import os
import re
import json
import random
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI  

import os
import re
import json
import random
from datasets import load_dataset
from openai import OpenAI

OPENAI_API_KEY = os.getenv("MY_API_KEY")

if not OPENAI_API_KEY:
    try:
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "key.env")
        with open(key_path, "r") as f:
            for line in f:
                if line.startswith("MY_API_KEY="):
                    OPENAI_API_KEY = line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass
OUTPUT_DIR = "data"
NUM_TO_AUGMENT = 1000
MAX_NUMBER_LIMIT = 1000
MAX_TEXT_LENGTH = 600

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def is_tiny_friendly(text):
    if re.search(r'\d+\.\d+', text): return False
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    if not numbers: return False
    if any(n > MAX_NUMBER_LIMIT for n in numbers): return False
    return True

def phase_1_split_data():
    print("PHASE 1: Splitting Data")
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    
    bucket_a = []
    bucket_b = []

    print(f"Scanning {len(dataset)} items...")
    for item in dataset:
        full_text = f"{item['question']} {item['answer']}"
        if is_tiny_friendly(full_text) and len(full_text) < MAX_TEXT_LENGTH:
            bucket_a.append(item)
        else:
            bucket_b.append(item)

    print(f"Bucket A: {len(bucket_a)} | Bucket B: {len(bucket_b)}")
    return bucket_a, bucket_b


async def simplify_single_problem(sem, item, max_retries=5):
    """
    Async worker for a single problem.
    Uses a Semaphore to control how many requests hit the API at once.
    Includes retry logic with exponential backoff for rate limits.
    """
    system_prompt = """
    You are a data generator for a tiny AI model. 
    Rewrite the user's math problem to be simpler.
    Constraints:
    1. Use ONLY integers between 0 and 100.
    2. NO decimals.
    3. Keep the text under 40 words.
    4. Return valid JSON with EXACTLY these three keys:
       - "question": the simplified math problem
       - "thinking": the step-by-step logic to solve it
       - "answer": the final numerical answer
    """
    user_prompt = f"Original: {item['question']}\nAns: {item['answer']}\n\nSimplify."

    async with sem:  # Waits here if too many concurrent requests are active
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-5-nano", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                print(".", end="", flush=True)  # Success indicator
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    print(f"x", end="", flush=True)
                    return None

async def phase_2_augment_data_async(bucket_b):
    print("\n PHASE 2: Async Augmentation")
    
    sample_to_process = random.sample(bucket_b, min(len(bucket_b), NUM_TO_AUGMENT))
    total = len(sample_to_process)
    print(f"Queueing {total} requests...")

    sem = asyncio.Semaphore(50)  # Balance speed vs rate limits (500 RPM max)
    counter = {"done": 0, "success": 0}
    lock = asyncio.Lock()

    async def wrapped_task(item):
        result = await simplify_single_problem(sem, item)
        async with lock:
            counter["done"] += 1
            if result:
                counter["success"] += 1
            if counter["done"] % 100 == 0:
                print(f"\n[{counter['done']}/{total}] ({counter['success']} success)", flush=True)
        return result

    tasks = [wrapped_task(item) for item in sample_to_process]
    results = await asyncio.gather(*tasks)
    
    bucket_c = [r for r in results if r]
    
    print(f"\nGenerated {len(bucket_c)} synthetic examples.")
    return bucket_c

def phase_3_verify_data(bucket_c):
    print(f"\nPHASE 3: Verifying {len(bucket_c)} augmented samples...")
    verified_bucket = []
    for item in bucket_c:
        if item.get('question') and item.get('answer') and item.get('thinking'):
            verified_bucket.append(item)
        else:
            print(f"  [REJECTED] Missing fields. Keys found: {list(item.keys())}")
    print(f"Verified: {len(verified_bucket)} / {len(bucket_c)}")
    return verified_bucket

def phase_4_final_assembly(bucket_a, verified_bucket_c):
    print("\nPHASE 4: Final Formatting")
    final_dataset = []
    for item in bucket_a:
        final_dataset.append(f"[BOS] Q: {item['question']} A: {item['answer']} [EOS]")
    for item in verified_bucket_c:
        final_dataset.append(f"[BOS] Q: {item['question']} Thinking: {item['thinking']} A: {item['answer']} [EOS]")
    
    random.shuffle(final_dataset)
    with open(f"{OUTPUT_DIR}/final_train_data.txt", "w") as f:
        for line in final_dataset:
            f.write(line.replace('\n', ' ') + '\n')
    print(f"Total samples in final_train_data.txt: {len(final_dataset)}")
    print("Done.")

async def main():
    bucket_a, bucket_b = phase_1_split_data()
    
    if bucket_b:
        bucket_c = await phase_2_augment_data_async(bucket_b)
    else:
        bucket_c = []
        
    verified_c = phase_3_verify_data(bucket_c)
    
    phase_4_final_assembly(bucket_a, verified_c)

if __name__ == "__main__":
    asyncio.run(main())