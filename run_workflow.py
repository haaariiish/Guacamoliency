from snakemake import snakemake

success = snakemake(
    snakefile="workflow.smk",          # chemin vers ton Snakefile
    cores=20,                        # nombre de cœurs à utiliser
    workdir=".",                    # répertoire de travail
    use_conda=False,                # active conda si besoin
    verbose=True,                   # verbosité
    dryrun=True                    # True pour tester sans exécuter
)

if success:
    print("Pipeline exécuté avec succès.")
else:
    print("Erreur dans le pipeline.")
