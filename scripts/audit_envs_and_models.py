import sys
import subprocess
import os

out_file = r"C:\PROJECTS\multimodal-stock-ai\reports\audit_envs_output.txt"
with open(out_file, "w", encoding="utf-8") as out:
    def log(msg):
        print(msg)
        out.write(msg + "\n")

    log("Auditing Python environments...")

    venvs = {
        "multimodal-stock-ai": r"C:\PROJECTS\multimodal-stock-ai\.venv\Scripts\python.exe",
        "finbert-lstm": r"C:\PROJECTS\Research paer\finbert-lstm\.venv\Scripts\python.exe"
    }

    code = """
import sys
modules = ['torch', 'tensorflow', 'keras', 'transformers', 'huggingface_hub', 'sklearn', 'pandas', 'numpy']
res = {}
for m in modules:
    try:
        mod = __import__(m)
        res[m] = getattr(mod, '__version__', 'ok')
    except ImportError as e:
        res[m] = f'NOT INSTALLED ({e})'
for k, v in res.items():
    print(f'  {k:<16}: {v}')
"""

    for name, path in venvs.items():
        log(f"\nEnvironment: {name} ({path})")
        if os.path.exists(path):
            res = subprocess.run([path, "-c", code], capture_output=True, text=True)
            log(res.stdout)
            if res.stderr:
                log("STDERR: " + res.stderr.strip()[:300])
        else:
            log("  Path does not exist!")

print(f"Audit written to {out_file}")
