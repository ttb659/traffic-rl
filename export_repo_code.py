import os

OUTPUT_FILE = "FULL_REPO_DUMP.txt"

INCLUDE_EXTENSIONS = {".py", ".xml", ".md", ".txt"}
EXCLUDE_DIRS = {".git", "__pycache__", ".idea", ".vscode", ".pytest_cache"}

def should_include_file(filename):
    return any(filename.endswith(ext) for ext in INCLUDE_EXTENSIONS)

def export_repo(root_dir="."):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(root_dir):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in sorted(files):
                if should_include_file(file):
                    filepath = os.path.join(root, file)

                    out.write("\n" + "=" * 80 + "\n")
                    out.write(f"FILE: {filepath}\n")
                    out.write("=" * 80 + "\n\n")

                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"[ERROR READING FILE] {e}\n")

    print(f"\n✅ Repository export completed: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_repo()
