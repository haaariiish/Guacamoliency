import subprocess
import sys

result = subprocess.run([
    sys.executable, 
    "-m", "snakemake",
    "-s", "workflow.smk",
    "--cores", "20",
    "--verbose",
    "--rerun-incomplete"
])

if result.returncode == 0:
    print("Pipeline exécuté avec succès.")
else:
    print("Erreur dans le pipeline.")