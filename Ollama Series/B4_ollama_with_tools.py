import pandas as pd
from pathlib import Path
import ollama

file_path = Path("enter_csv_file_path_here")

if not file_path.exists():
    print("File does not exist.")
    exit()

df = pd.read_csv(file_path)

def check_item(item):
    if item in df["item"].values:
        row = df.loc[df["item"] == item].iloc[0]
        return {
            "item": item,
            "available": True,
            "quantity": int(row["quantity"]),
            "price": float(row["price"])
        }
    return {
        "item": item,
        "available": False
    }

def update_quantity(item, quantity):
    global df
    if item in df["item"].values:
        df.loc[df["item"] == item, "quantity"] = quantity
        return f"Updated {item} quantity to {quantity}."
    return f"{item} is not available."

def update_price(item, price):
    global df
    if item in df["item"].values:
        df.loc[df["item"] == item, "price"] = price
        return f"Updated {item} price to {price}."
    return f"{item} is not available."

def add_item(item, quantity, price):
    global df
    if item in df["item"].values:
        return f"{item} already exists."
    new_row = {"item": item, "quantity": quantity, "price": price}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return f"Added {item} to the inventory."

def save_changes():
    df.to_csv(file_path, index=False)
    return "Changes saved to file."

available_functions = {
    "check_item": check_item,
    "add_item": add_item,
    "update_quantity": update_quantity,
    "update_price": update_price,
    "save_changes": save_changes
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_item",
            "description": "Check if an item is available in the inventory and return its details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The name of the item to check."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add a new item to the inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number"}
                },
                "required": ["item", "quantity", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_quantity",
            "description": "Update the quantity of an existing item in the inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"type": "integer"}
                },
                "required": ["item", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_price",
            "description": "Update the price of an existing item in the inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "price": {"type": "number"}
                },
                "required": ["item", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_changes",
            "description": "Save all current inventory changes to the CSV file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

user_input = input("You: ")
messages = [
    {"role": "user", "content": user_input}
]

response = ollama.chat(
    model="qwen3.5:9b",
    messages=messages,
    tools=tools
)

while True: 
    messages.append(response["message"])

    tool_calls = response["message"].get("tool_calls", [])

    if not tool_calls:
        break

    for call in tool_calls:
        function_name = call["function"]["name"]
        arguments = call["function"]["arguments"]

        if function_name in available_functions:
            try:
                result = available_functions[function_name](**arguments)
            except Exception as e:
                result = f"Error while executing {function_name}: {e}"
        else:
            result = f"Function {function_name} is not available."
        
        messages.append({
            "role": "tool",
            "name": function_name,
            "content": str(result)
        })

    response = ollama.chat(
        model="qwen3.5:9b",
        messages=messages,
        tools=tools
    )

print(f"Response: {response['message']['content']}")