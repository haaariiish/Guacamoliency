from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

#from guacamoliency.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    #input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    #output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

   return
  
if __name__ == "__main__":
    app()
