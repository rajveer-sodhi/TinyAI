
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
OUTPUT_DIR = "preprocessing/data"
NUM_TO_AUGMENT = 10
MAX_NUMBER_LIMIT = 1000
MAX_TEXT_LENGTH = 500

# Dataset selection: Set to True to include GSM8K, False to use only Orca Math
USE_GSM8K = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_final_answer(answer_text):
    """
    Extract just the final numeric answer from an answer string.
    Handles formats like:
    - "Step by step explanation... #### 42" -> "42"
    - "42" -> "42"
    - "The answer is 42." -> "42"
    """
    # Check if answer contains "####" separator (GSM8K and Orca Math format)
    if "####" in answer_text:
        
        final_answer = answer_text.split("####")[-1].strip()
        return final_answer
    
    # Otherwise, try to extract the last number from the text
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return numbers[-1]
    
    # If no number found, return the answer as-is (shouldn't happen with math problems)
    return answer_text.strip()

# Only initialize OpenAI client if we might use it (augmentation is disabled but might be re-enabled)
# If using GSM8K only, we don't need the OpenAI client
client = None
if not USE_GSM8K and OPENAI_API_KEY:
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


def load_gsm8k():
    """
    Load GSM8K dataset in its entirety, no filtering or augmentation.
    Returns list of items with 'question' and 'answer' fields.
    Extracts only the final numeric answer (after ####).
    """
    print("\nPHASE 1.5: Loading GSM8K Dataset")
    try:
        # GSM8K has 'train' and 'test' splits, we'll use 'train' for training data
        gsm8k_train = load_dataset("gsm8k", "main", split="train")
        print(f"Loaded {len(gsm8k_train)} GSM8K training samples")
        
        # Convert to list format compatible with our pipeline
        gsm8k_items = []
        for item in gsm8k_train:
            # GSM8K format: {'question': str, 'answer': str}
            # Extract only the final answer (after ####)
            final_answer = extract_final_answer(item['answer'])
            gsm8k_items.append({
                'question': item['question'],
                'answer': final_answer
            })
        
        print(f"GSM8K: {len(gsm8k_items)} samples ready")
        return gsm8k_items
    except Exception as e:
        print(f"Warning: Failed to load GSM8K dataset: {e}")
        return []


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

def phase_4_final_assembly(bucket_a, bucket_b, verified_bucket_c, gsm8k_items=None):
    print("\nPHASE 4: Final Formatting")
    
    final_dataset = []
    
    # If using GSM8K only, skip Orca Math buckets
    if gsm8k_items:
        # Add GSM8K samples only (answer already extracted in load_gsm8k)
        for item in gsm8k_items:
            final_dataset.append(f"[BOS] Q: {item['question']} A: {item['answer']} [EOS]")
        print(f"GSM8K samples: {len(gsm8k_items)}")
    else:
        # Use Orca Math buckets
        # Limit to 10,000 samples from each bucket
        MAX_SAMPLES_PER_BUCKET = 10000
        
        # Sample from bucket A
        bucket_a_sample = random.sample(bucket_a, min(len(bucket_a), MAX_SAMPLES_PER_BUCKET))
        
        # Sample from bucket B
        bucket_b_sample = random.sample(bucket_b, min(len(bucket_b), MAX_SAMPLES_PER_BUCKET))
        
        # Add bucket A samples (simple problems) - extract final answer only
        for item in bucket_a_sample:
            final_answer = extract_final_answer(item['answer'])
            final_dataset.append(f"[BOS] Q: {item['question']} A: {final_answer} [EOS]")
        
        # Add bucket B samples (complex problems) - extract final answer only
        for item in bucket_b_sample:
            final_answer = extract_final_answer(item['answer'])
            final_dataset.append(f"[BOS] Q: {item['question']} A: {final_answer} [EOS]")
        
        print(f"Bucket A: {len(bucket_a)} available, {len(bucket_a_sample)} used")
        print(f"Bucket B: {len(bucket_b)} available, {len(bucket_b_sample)} used")
    
    # Add augmented bucket C (if any) - extract final answer only
    for item in verified_bucket_c:
        final_answer = extract_final_answer(item['answer'])
        final_dataset.append(f"[BOS] Q: {item['question']} Thinking: {item['thinking']} A: {final_answer} [EOS]")
    
    random.shuffle(final_dataset)
    with open(f"{OUTPUT_DIR}/final_train_data.txt", "w", encoding="utf-8") as f:
        for line in final_dataset:
            f.write(line.replace('\n', ' ') + '\n')
    
    if verified_bucket_c:
        print(f"Augmented samples: {len(verified_bucket_c)}")
    print(f"Total samples in final_train_data.txt: {len(final_dataset)}")
    print("Done.")

async def main():
    # Load GSM8K if enabled, otherwise use Orca Math
    gsm8k_items = None
    bucket_a = []
    bucket_b = []
    
    if USE_GSM8K:
        # Use GSM8K only
        gsm8k_items = load_gsm8k()
    else:
        # Use Orca Math
        bucket_a, bucket_b = phase_1_split_data()
    
    # if bucket_b:
    #     bucket_c = await phase_2_augment_data_async(bucket_b)
    # else:
    #     bucket_c = []

    bucket_c = []
        
    verified_c = phase_3_verify_data(bucket_c)
    
    phase_4_final_assembly(bucket_a, bucket_b, verified_c, gsm8k_items)

if __name__ == "__main__":
    asyncio.run(main())