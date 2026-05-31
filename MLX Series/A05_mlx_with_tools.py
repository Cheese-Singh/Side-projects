import json
from pathlib import Path
import pandas as pd
from mlx_vlm import load, generate

llm_model = None
llm_processor = None

tool_add_tasks_to_db = {
    "type": "function",
    "function": {
        "name": "add_task",
        "description": "Add a new task to the task database.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The text of the task to add."
                }
            },
            "required": ["task"]
        }
    }
}
tool_create_file = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a new file with the specified name and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the file to create."
                },
                "content": {
                    "type": "string",
                    "description": "The content of the file to create."
                }
            },
            "required": ["name", "content"]
        }
    }
}
tool_read_file = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a file with the specified name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the file to read."
                }
            },
            "required": ["name"]
        }
    }
}
tool_delete_file = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Delete a file with the specified name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the file to delete."
                }
            },
            "required": ["name"]
        }
    }
}

tools = [tool_add_tasks_to_db, tool_create_file, tool_read_file, tool_delete_file]

tool_map = {tool["function"]["name"]: tool for tool in tools}

TEMP_TOOL_FILE = Path("tool_check_file.txt")

BASE_DIR = Path(__file__).parent.resolve()

TEMP_TOOL_FILE = BASE_DIR / "tool_check_file.txt"

TASKS_FILE = BASE_DIR / "tasks_db.csv"

def _load_tasks() -> pd.DataFrame:
    if TASKS_FILE.exists():
        try:
            return pd.read_csv(TASKS_FILE, parse_dates=["Start-date", "Completion-date"])  # type: ignore
        except Exception:
            return pd.DataFrame(columns=["Task", "Status", "Start-date", "Completion-date"])
    return pd.DataFrame(columns=["Task", "Status", "Start-date", "Completion-date"])

tasks_db = _load_tasks()

def _persist_tasks() -> None:
    try:
        tasks_db.to_csv(TASKS_FILE, index=False)
    except Exception:
        pass

def add_task(task: str) -> str:
    """Add a new task and persist it immediately."""
    global tasks_db
    if task in tasks_db["Task"].values:
        return f"Task '{task}' already exists in the database."
    new_task = {
        "Task": task,
        "Status": "Pending",
        "Start-date": pd.Timestamp.now().isoformat(),
        "Completion-date": None,
    }
    tasks_db = pd.concat([tasks_db, pd.DataFrame([new_task])], ignore_index=True)
    _persist_tasks()
    return f"Added task: {task}"

def list_tasks() -> list:
    return tasks_db.fillna("").to_dict(orient="records")

def create_file(name: str, content: str) -> str:
    try:
        candidate = (BASE_DIR / name).resolve()
    except Exception:
        return f"Invalid filename: {name}"
    try:
        candidate.relative_to(BASE_DIR)
    except Exception:
        return f"Permission denied: cannot create file outside sandbox"
    file_path = candidate
    if file_path.exists():
        return f"File '{name}' already exists."
    try:
        file_path.write_text(content, encoding="utf-8")
        return f"Created file '{name}'."
    except Exception as exc:
        return f"Failed to create file '{name}': {exc}"

def read_file(name: str) -> str:
    try:
        candidate = (BASE_DIR / name).resolve()
    except Exception:
        return f"Invalid filename: {name}"
    try:
        candidate.relative_to(BASE_DIR)
    except Exception:
        return f"Permission denied: cannot read file outside sandbox"
    file_path = candidate
    if not file_path.exists():
        return f"File '{name}' does not exist."
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Failed to read file '{name}': {exc}"

def delete_file(name: str) -> str:
    try:
        candidate = (BASE_DIR / name).resolve()
    except Exception:
        return f"Invalid filename: {name}"
    try:
        candidate.relative_to(BASE_DIR)
    except Exception:
        return f"Permission denied: cannot delete file outside sandbox"
    file_path = candidate
    if not file_path.exists():
        return f"File '{name}' does not exist."
    try:
        file_path.unlink()
        return f"Deleted file '{name}'."
    except Exception as exc:
        return f"Failed to delete file '{name}': {exc}"

def _find_first_json(text: str):
    stack = []
    start_idx = None
    for idx, char in enumerate(text):
        if char == "{":
            if start_idx is None:
                start_idx = idx
            stack.append(char)
        elif char == "}" and stack:
            stack.pop()
            if not stack and start_idx is not None:
                yield text[start_idx : idx + 1]
                start_idx = None

def extract_json_object(text: str):
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    for candidate in _find_first_json(text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

def _strip_assistant_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in ("assistant:", "assistant", "system:", "system"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break
    return stripped

def _extract_pure_json(text: str):
    text = _strip_assistant_prefix(text)
    for candidate in _find_first_json(text):
        before = text[: text.index(candidate)].strip()
        after = text[text.index(candidate) + len(candidate) :].strip()
        if before == "" and after == "":
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None

def parse_simple_command(text: str):
    """Fallback simple command parser for common one-line instructions.

    Returns a tool call dict {'name': ..., 'arguments': {...}} or None.
    """
    s = text.strip().lower()

    import re
    m = re.search(r"create\s+a\s+file(?:\s+named)?\s+['\"]?([^'\"\s]+)['\"]?\s+with\s+(?:the\s+)?content[:]?\s*['\"]?([^'\"]+)['\"]?", text, flags=re.I | re.S)
    if m:
        name = m.group(1).strip()
        content = m.group(2).strip()
        return {"name": "create_file", "arguments": {"name": name, "content": content}}

    m = re.search(r"read\s+(?:the\s+)?(?:content of|contents of|file)\s+['\"]?([^'\"\s]+)['\"]?", text, flags=re.I)
    if m:
        return {"name": "read_file", "arguments": {"name": m.group(1).strip()}}

    m = re.search(r"delete\s+(?:the\s+)?file(?:\s+named)?\s+['\\\"]?([^'\\\"\s]+)['\\\"]?", text, flags=re.I)
    if m:
        return {"name": "delete_file", "arguments": {"name": m.group(1).strip()}}

    m = re.search(r"add\s+'([^']+)'\s+to (?:my )?(?:task list|task database)", text, flags=re.I)
    if m:
        return {"name": "add_task", "arguments": {"task": m.group(1)}}

    m = re.search(r"add the task ([^\.]+) to (?:my )?(?:task list|task database)", text, flags=re.I)
    if m:
        return {"name": "add_task", "arguments": {"task": m.group(1).strip()}}

    m = re.search(r"add a task to (?:the )?(?:task database|task list) with the text ['\"]?([^'\"]+)['\"]?", text, flags=re.I)
    if m:
        return {"name": "add_task", "arguments": {"task": m.group(1).strip()}}

    m = re.search(r"add a task with the text ['\"]?([^'\"]+)['\"]?", text, flags=re.I)
    if m:
        return {"name": "add_task", "arguments": {"task": m.group(1).strip()}}

    m = re.search(r"list\s+(?:all\s+)?tasks(?:\s+in\s+(?:the\s+)?(?:task\s+database|task\s+list))?", text, flags=re.I)
    if m:
        return {"name": "list_tasks", "arguments": {}}

    return None

def is_tool_prompt(prompt: str) -> bool:
    return parse_simple_command(prompt) is not None

def run_tool(tool_call: dict) -> str:
    if not isinstance(tool_call, dict):
        return "No valid tool call found."

    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if name == "add_task":
        task_text = arguments.get("task")
        if not task_text and isinstance(arguments.get("task"), dict):
            task_text = arguments["task"].get("name")
        if not task_text:
            return "Tool call missing required argument: task"
        return add_task(task_text)

    if name == "create_file":
        return create_file(arguments.get("name", ""), arguments.get("content", ""))

    if name == "read_file":
        return read_file(arguments.get("name", ""))

    if name == "delete_file":
        return delete_file(arguments.get("name", ""))

    if name == "list_tasks":
        return json.dumps(list_tasks(), default=str)

    return f"Tool {name} is not available."

def verify_tool(tool_name: str) -> str:
    sample_inputs = {
        "add_task": {"task": "Verify task add_tool"},
        "create_file": {"name": str(TEMP_TOOL_FILE), "content": "tool check content"},
        "read_file": {"name": str(TEMP_TOOL_FILE)},
        "delete_file": {"name": str(TEMP_TOOL_FILE)},
    }
    if tool_name not in sample_inputs:
        return f"No verification sample for tool {tool_name}."

    if tool_name in {"read_file", "delete_file"} and not TEMP_TOOL_FILE.exists():
        create_file(str(TEMP_TOOL_FILE), "tool check content")

    return run_tool({"name": tool_name, "arguments": sample_inputs[tool_name]})

def verify_all_tools() -> dict:
    results = {}
    for tool in tools:
        name = tool["function"]["name"]
        results[name] = verify_tool(name)
    return results

def ensure_model_loaded():
    global llm_model, llm_processor
    if llm_model is None or llm_processor is None:
        llm_model, llm_processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

def get_response(prompt: str) -> str:
    system_prompt = f"""
You are a helpful assistant that can use the following tools:
{json.dumps(tools, indent=2)}

If the user asks you to call a tool, output exactly one JSON object and nothing else.
The object must be in this format:
{{
  "name": "tool_name",
  "arguments": {{
    ...
  }}
}}

Use the tool name and arguments exactly as defined in the tool metadata.
Do not add any extra explanatory text or markdown when returning tool calls.
If the user asks a normal question, answer directly in plain text.
Do not include labels such as System, User, or Assistant in the response.
"""

    prompt_call = parse_simple_command(prompt)
    if prompt_call:
        return run_tool(prompt_call)

    full_prompt = f"{system_prompt}\n{prompt}\n"

    ensure_model_loaded()

    response = generate(
        llm_model,
        llm_processor,
        image=None,
        prompt=full_prompt,
        temperature=0.1,
        repetition_penalty=1.2,
        max_tokens=512,
        verbose=False,
    )

    response_text = response.text if hasattr(response, "text") else str(response)
    tool_call = _extract_pure_json(response_text)
    if tool_call and isinstance(tool_call, dict) and tool_call.get("name"):
        return run_tool(tool_call)

    return response_text

if __name__ == "__main__":
    print(get_response("Create a file named 'example.txt' with the content 'This is an example file.'"))
    print(get_response("Read the content of 'example.txt'."))
    print(get_response("Delete the file named 'example.txt'."))
    print(get_response("Add a task to the task database with the text 'Finish the project.'"))
    print(get_response("List all tasks in the task database."))
    print(get_response("Where is Mount Kilimanjaro located?"))