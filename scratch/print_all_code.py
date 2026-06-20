import json
import sys

def print_user_inputs(path):
    sys.stdout.reconfigure(encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('source') == 'USER_EXPLICIT':
                print(f"Step {data.get('step_index')}: {data.get('content')}")
                print("=" * 60)

if __name__ == "__main__":
    print_user_inputs("C:/Users/HenryHuang/.gemini/antigravity/brain/b3af31f2-be49-4e5f-9433-5d770711fa37/.system_generated/logs/transcript.jsonl")
