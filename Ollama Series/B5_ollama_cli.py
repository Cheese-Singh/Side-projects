import argparse
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
MODELFILE = BASE_DIR / "Modelfile"
PROMPTS = BASE_DIR / "test_prompts.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_NAME = "sentiment:latest"

def run_command(command: list[str], input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed or not available in PATH.")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n\n{error.stderr.strip()}"
        )

def validate_project() -> None:
    if not MODELFILE.exists():
        raise FileNotFoundError(f"Modelfile not found at: {MODELFILE}")

    OUTPUT_DIR.mkdir(exist_ok=True)

def model_exists() -> bool:
    output = run_command(["ollama", "list"])
    return any(line.split()[0] == MODEL_NAME for line in output.splitlines()[1:] if line.strip())

def create_model() -> None:
    validate_project()

    print(f"Creating Ollama model: {MODEL_NAME}")
    output = run_command(["ollama", "create", MODEL_NAME, "-f", str(MODELFILE)])

    if output:
        print(output)

    print(f"Model '{MODEL_NAME}' is ready.")

def ensure_model_ready() -> None:
    validate_project()

    if not model_exists():
        print(f"Model '{MODEL_NAME}' not found. Creating it now...")
        create_model()

def run_prompt(prompt: str) -> str:
    ensure_model_ready()

    return run_command(
        ["ollama", "run", MODEL_NAME],
        input_text=prompt
    )

def load_test_prompts() -> list[str]:
    if not PROMPTS.exists():
        raise FileNotFoundError(f"Prompts file not found at: {PROMPTS}")

    return [
        prompt.strip()
        for prompt in PROMPTS.read_text(encoding="utf-8").split("\n---\n")
        if prompt.strip()
    ]

def evaluate_prompts() -> None:
    ensure_model_ready()

    prompts = load_test_prompts()

    if not prompts:
        raise ValueError("No test prompts found in test_prompts.txt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"evaluation_report_{timestamp}.md"

    report = [
        "# Ollama Model Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Model: `{MODEL_NAME}`",
        "",
    ]

    for index, prompt in enumerate(prompts, start=1):
        print(f"Running test prompt {index}/{len(prompts)}...")

        response = run_prompt(prompt)

        report.extend([
            f"## Test {index}",
            "",
            "### Prompt",
            "```text",
            prompt,
            "```",
            "",
            "### Response",
            "```text",
            response,
            "```",
            ""
        ])

    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Evaluation report saved to: {report_path}")

def chat() -> None:
    ensure_model_ready()

    print(f"Chat started with model '{MODEL_NAME}'. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        if not user_input:
            continue

        response = run_prompt(user_input)
        print(f"\n{MODEL_NAME}:\n{response}\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage, test, and chat with a local Ollama Modelfile."
    )

    parser.add_argument(
        "command",
        choices=["validate", "create", "evaluate", "chat"],
        help="Command to run."
    )

    args = parser.parse_args()

    if args.command == "validate":
        validate_project()
        print("Project validation successful.")

    elif args.command == "create":
        create_model()

    elif args.command == "evaluate":
        evaluate_prompts()

    elif args.command == "chat":
        chat()

if __name__ == "__main__":
    main()