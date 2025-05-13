import os
# ──────── Suppress TensorFlow Logs ────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm
from Bio import SeqIO
import warnings
import logging
import sys
import argparse
import os

warnings.filterwarnings("ignore")

# ──────── Set Device ────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────── Load Tokenizer and Model from Hugging Face ────────
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
esm_model.eval()

# ──────── Local Path Setup ────────
base_dir = os.path.dirname(__file__)
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
classifier_path = os.path.join(base_dir, "models", "epitope_classifier.pth")

# ──────── Load Scaler and Classifier ────────
scaler = joblib.load(scaler_path)

class EpitopeClassifier(torch.nn.Module):
    def __init__(self):
        super(EpitopeClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

model = EpitopeClassifier().to(device)
model.load_state_dict(torch.load(classifier_path, map_location=device))
model.eval()

# ──────── Embedding Function ────────
def get_esm_embedding(peptide):
    try:
        inputs = tokenizer(peptide, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = esm_model(**inputs)
        token_embeddings = output.last_hidden_state
        return token_embeddings[0, 1:-1].mean(dim=0).cpu().numpy()
    except Exception as e:
        logging.warning(f"[ERROR] {peptide}: {e}")
        return None

# ──────── Peptide Window Generator ────────
def extract_peptides(sequence, min_len=8, max_len=25, full=False):
    if full:
        return [(1, len(sequence), sequence)]
    return [(i + 1, i + l, sequence[i:i + l])
            for l in range(min_len, max_len + 1)
            for i in range(len(sequence) - l + 1)]

# ──────── Prediction Core ────────
def run_prediction(args):
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO if args.verbose else logging.WARNING, format='%(message)s')

    log = logging.getLogger()

    records = list(SeqIO.parse(args.fasta, "fasta"))
    all_results = []
    top_n_results = []

    log.info(f"📥 Loaded {len(records)} sequences from {args.fasta}")
    log.info(f"🔍 Full-length: {args.use_full_length}, Window: {args.min_len}-{args.max_len}")
    log.info(f"⚙️ Threshold: {args.threshold}, Top N: {args.top_n}, Output: {'CSV' if args.save_csv else 'Excel'}")

    for record in records:
        header = record.id
        seq = str(record.seq).upper()
        seq = "".join(filter(str.isalpha, seq))
        if not seq or not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            log.warning(f"[SKIPPED] {header}: Invalid sequence")
            continue

        log.info(f"🧬 Processing {header} (Length: {len(seq)})")
        peptides = extract_peptides(seq, min_len=args.min_len, max_len=args.max_len, full=args.use_full_length)

        temp = []
        for start, end, pep in tqdm(peptides, desc=f"Embedding {header}"):
            emb = get_esm_embedding(pep)
            if emb is not None:
                temp.append({
                    "Header": header,
                    "Start": start,
                    "End": end,
                    "Length": end - start + 1,
                    "Peptide": pep,
                    "Embedding": emb
                })

        if not temp:
            continue

        X = scaler.transform([x["Embedding"] for x in temp])
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), args.batch_size):
                batch = X_tensor[i:i+args.batch_size]
                prob = torch.sigmoid(model(batch)).cpu().numpy().flatten()
                preds.extend(prob)

        for i, prob in enumerate(preds):
            temp[i]["Probability"] = prob
            del temp[i]["Embedding"]

        df = pd.DataFrame(temp)
        df = df[df["Probability"] >= args.threshold]
        df.sort_values(by="Probability", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Rank"] = df.index + 1
        df = df[["Header", "Rank", "Start", "End", "Length", "Peptide", "Probability"]]

        all_results.append(df)
        top_n_results.append(df.head(args.top_n))

    # ──────── Save Outputs ────────
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_top = pd.concat(top_n_results, ignore_index=True)

        if args.save_csv:
            df_all.to_csv(f"{args.output_prefix}_all.csv", index=False)
            df_top.to_csv(f"{args.output_prefix}_top{args.top_n}.csv", index=False)
        else:
            df_all.to_excel(f"{args.output_prefix}_all.xlsx", index=False)
            df_top.to_excel(f"{args.output_prefix}_top{args.top_n}.xlsx", index=False)

        log.info(f"✅ Saved all predictions to {args.output_prefix}_all.{'csv' if args.save_csv else 'xlsx'}")
        log.info(f"✅ Saved top {args.top_n} predictions to {args.output_prefix}_top{args.top_n}.{'csv' if args.save_csv else 'xlsx'}")
    else:
        log.warning("❌ No valid sequences processed.")

# ──────── CLI Entrypoint ────────
def main():
    parser = argparse.ArgumentParser(description="Predict B cell epitopes from multi-FASTA input.")
    parser.add_argument("fasta", type=str, help="Multi-FASTA file with protein sequences")
    parser.add_argument("--min_len", type=int, default=8, help="Minimum epitope length")
    parser.add_argument("--max_len", type=int, default=25, help="Maximum epitope length")
    parser.add_argument("--use_full_length", action="store_true", help="Use full-length protein as a single peptide")
    parser.add_argument("--threshold", type=float, default=0.0, help="Probability threshold for filtering")
    parser.add_argument("--top_n", type=int, default=10, help="Top N peptides per sequence")
    parser.add_argument("--output_prefix", type=str, default="epitope_predictions", help="Output file prefix")
    parser.add_argument("--save_csv", action="store_true", help="Save results in CSV format instead of Excel")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log_file", type=str, help="Optional log file path")
    args = parser.parse_args()
    run_prediction(args)

