# token_count.py
import argparse
from transformers import AutoTokenizer

def count_tokens(text: str, tokenizer) -> int:
    # add_special_tokens=False so you count just the string itself
    return len(tokenizer.encode(text, add_special_tokens=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--text", default=None, help="Text to count tokens for.")
    ap.add_argument("--file", default=None, help="Path to a text file to count tokens for.")
    args = ap.parse_args()

    if (args.text is None) == (args.file is None):
        raise SystemExit("Provide exactly one of --text or --file")

    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    tok = AutoTokenizer.from_pretrained(args.model_id)
    n = count_tokens(text, tok)
    print(n)

if __name__ == "__main__":
    main()
