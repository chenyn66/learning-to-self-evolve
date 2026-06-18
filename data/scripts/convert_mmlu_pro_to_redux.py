import random
import datasets
from datasets import Dataset, DatasetDict, Features, Value, Sequence

# Set seed for reproducibility
random.seed(42)

def transform_item(item):
    question = item['question']
    options = item['options']
    answer_index = item['answer_index']
    
    # Validation
    if answer_index is None or not isinstance(answer_index, int):
        return None
    if answer_index < 0 or answer_index >= len(options):
        return None
        
    correct_option = options[answer_index]
    
    # Get distractors
    distractors = [opt for i, opt in enumerate(options) if i != answer_index]
    
    # Select 3 distractors to make total 4 options
    # If fewer than 3 distractors, use all of them
    k = min(len(distractors), 3)
    selected_distractors = random.sample(distractors, k)
    
    # Combine
    choices = [correct_option] + selected_distractors
    random.shuffle(choices)
    
    # Find new answer index
    try:
        new_answer_index = choices.index(correct_option)
    except ValueError:
        return None
        
    return {
        'question': question,
        'choices': choices,
        'answer': new_answer_index,
        'error_type': '',
        'source': item.get('src', ''), 
        'correct_answer': str(new_answer_index),
        'potential_reason': '',
        'category': item['category'] # Temporary for grouping
    }

def main():
    print("Loading MMLU-Pro...")
    # Using trust_remote_code=True as seen in inspection
    pro_dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    
    # Filter 'ori_mmlu'
    print("Filtering out 'ori_mmlu' sources...")
    # Filter logic: keep if NOT starting with 'ori_mmlu'
    filtered_pro = pro_dataset.filter(lambda x: x['src'] is not None and not x['src'].startswith('ori_mmlu'))
    print(f"Filtered {len(pro_dataset)} -> {len(filtered_pro)} items.")
    
    # Transform items
    print("Transforming items...")
    
    transformed_items_by_category = {}
    
    # Define features for Redux format
    features = Features({
        'question': Value('string'),
        'choices': Sequence(Value('string')),
        'answer': Value('int64'),
        'error_type': Value('string'),
        'source': Value('string'),
        'correct_answer': Value('string'),
        'potential_reason': Value('string')
    })
    
    for item in filtered_pro:
        t_item = transform_item(item)
        if t_item is None:
            continue
            
        cat = t_item['category']
        if not cat:
            cat = "uncategorized"
            
        final_item = {k: v for k, v in t_item.items() if k != 'category'}
        
        if cat not in transformed_items_by_category:
            transformed_items_by_category[cat] = []
        transformed_items_by_category[cat].append(final_item)
        
    # Create DatasetDict
    ds_dict = {}
    for cat, items in transformed_items_by_category.items():
        print(f"Category: {cat}, Count: {len(items)}")
        ds = Dataset.from_list(items, features=features)
        ds_dict[cat] = ds
        
    final_dataset = DatasetDict(ds_dict)
    
    output_path = "data/mmlu-pro-redux"
    print(f"Saving to {output_path}...")
    final_dataset.save_to_disk(output_path)
    print("Done.")

if __name__ == "__main__":
    main()

