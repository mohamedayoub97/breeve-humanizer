"""
app.py
------
Web version of the Breeve Cleaning & Humanizer tool, built with Streamlit
so it can be hosted and shared as a link (e.g. on Streamlit Community Cloud).

The OpenRouter API key is NEVER shown to or editable by the person using the
app: it's read only from Streamlit secrets (st.secrets["OPENROUTER_API_KEY"]),
which is set once by the app owner in the hosting dashboard and never exposed
in the UI or in the source code.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from humanizer import Humanizer, HumanizerConfig
from openrouter_client import OpenRouterClient
from pipeline import Pipeline, PipelineResult
from pricing import fetch_model_pricing, estimate_cost_usd

# ---------------------------------------------------------------------------
# Page & API key setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Breeve — Cleaning & Humanizer", page_icon="💬", layout="wide")

API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

DEFAULT_MODEL = "openai/gpt-4o-mini"
SUGGESTED_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku",
    "mistralai/mistral-small-3.1-24b-instruct",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.1-8b-instruct",
]

INPUT_COLUMN_CANDIDATES = ["input brut", "input_brut", "input", "raw", "raw input", "message brut"]


def _normalize_col_name(name: str) -> str:
    import re
    import unicodedata
    name = str(name).replace("\ufeff", "")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"\s+", " ", name, flags=re.UNICODE).strip()
    return name.lower()


if not API_KEY:
    st.error(
        "Aucune clé OpenRouter configurée côté serveur. "
        "L'administrateur doit ajouter OPENROUTER_API_KEY dans les secrets de l'app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Persistence for the "Message unique" conversation history
#
# st.session_state only lives for as long as the Streamlit process stays up.
# If the app restarts (redeploy, host waking the app up from sleep, a crash),
# session_state is wiped and every past exchange disappears. To actually keep
# the conversation around, we mirror it to a small JSON file on disk and
# reload it the next time the app starts.
#
# Note: this file lives on the app's local disk. On hosts with an ephemeral
# filesystem (e.g. a fresh redeploy from git on Streamlit Community Cloud),
# the file itself can be wiped even though this code will happily recreate
# it. Waking up from inactivity sleep on the same deployment does not wipe
# it. If you need history to survive redeploys too, or need it kept
# separately per user, it should go to a real database instead of a local
# file — happy to wire that up if that's what you need.
# ---------------------------------------------------------------------------

HISTORY_FILE = "conversation_history.json"


def _serialize_result(result: "PipelineResult") -> dict:
    return {
        "message": result.message,
        "metadata": result.metadata,
        "humanized_message": result.humanized_message,
        "applied_rules": list(result.applied_rules or []),
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
        "completion_tokens": result.completion_tokens,
        "error": result.error,
    }


def _deserialize_result(data: dict) -> SimpleNamespace:
    # A lightweight stand-in for PipelineResult, carrying just the attributes
    # render_exchange() actually reads. Avoids depending on PipelineResult's
    # exact constructor signature when rebuilding history from disk.
    return SimpleNamespace(
        message=data.get("message", ""),
        metadata=data.get("metadata", ""),
        humanized_message=data.get("humanized_message", ""),
        applied_rules=data.get("applied_rules", []),
        latency_ms=data.get("latency_ms", 0.0),
        total_tokens=data.get("total_tokens"),
        completion_tokens=data.get("completion_tokens"),
        error=data.get("error"),
    )


def _load_conversation() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        return [(item["raw_input"], _deserialize_result(item["result"])) for item in raw_items]
    except Exception:
        # Corrupted or unreadable history file: start fresh instead of
        # crashing the whole app on load.
        return []


def _save_conversation(conversation: list) -> None:
    try:
        payload = [
            {"raw_input": raw, "result": _serialize_result(result)}
            for raw, result in conversation
        ]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Persistence is a nice-to-have; never let a save failure break the UI.
        pass


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.header("⚙ Configuration")

model = st.sidebar.selectbox("Modèle (Cleaning)", SUGGESTED_MODELS, index=0)

st.sidebar.subheader("Humanizer — règles")

col1, col2 = st.sidebar.columns(2)
enable_r01 = col1.checkbox("R-01 Abréviation", value=True)
prob_r01 = col2.slider("Prob.", 0.0, 1.0, 0.5, key="p01", label_visibility="collapsed")

col1, col2 = st.sidebar.columns(2)
enable_r02 = col1.checkbox("R-02 Accent", value=True)
prob_r02 = col2.slider("Prob.", 0.0, 1.0, 0.5, key="p02", label_visibility="collapsed")

col1, col2 = st.sidebar.columns(2)
enable_r03 = col1.checkbox("R-03 Omission lettre", value=True)
prob_r03 = col2.slider("Prob.", 0.0, 1.0, 0.5, key="p03", label_visibility="collapsed")
st.sidebar.caption("⚠ R-03 exclusif avec R-04")

col1, col2 = st.sidebar.columns(2)
enable_r04 = col1.checkbox("R-04 Fusion mots", value=True)
prob_r04 = col2.slider("Prob.", 0.0, 1.0, 0.5, key="p04", label_visibility="collapsed")
st.sidebar.caption("⚠ R-04 exclusif avec R-03")

col1, col2 = st.sidebar.columns(2)
enable_r05 = col1.checkbox("R-05 Ponctuation", value=True)
prob_r05 = col2.slider("Prob.", 0.0, 1.0, 0.5, key="p05", label_visibility="collapsed")

weight_r03 = st.sidebar.slider("Poids R-03 vs R-04 (quand les deux sont actives)", 0.0, 1.0, 0.5)

seed_raw = st.sidebar.text_input("Seed (optionnel, vide = aléatoire par message)")
seed = int(seed_raw) if seed_raw.strip().isdigit() else None


def build_humanizer_config() -> HumanizerConfig:
    return HumanizerConfig(
        enable_r01=enable_r01, enable_r02=enable_r02, enable_r03=enable_r03,
        enable_r04=enable_r04, enable_r05=enable_r05,
        prob_r01_abbreviation=prob_r01, prob_r02_accent=prob_r02,
        prob_r03_letter_omission=prob_r03, prob_r04_word_fusion=prob_r04,
        prob_r05_punctuation=prob_r05,
        weight_r03=weight_r03, weight_r04=1 - weight_r03,
        rng_seed=seed,
    )


def build_pipeline() -> Pipeline:
    client = OpenRouterClient(api_key=API_KEY, model=model)
    humanizer = Humanizer(build_humanizer_config())
    return Pipeline(client, humanizer)


# ---------------------------------------------------------------------------
# Main area — tabs for single message vs batch CSV
# ---------------------------------------------------------------------------

st.title("💬 Cleaning → Humanizer")

tab_single, tab_batch = st.tabs(["Message unique", "Traitement par lot (CSV)"])

# -- Single message tab -----------------------------------------------------

RESULT_CARD_CSS = """
<style>
.result-card { background:#e5e7eb; border-radius:12px; padding:16px 18px; margin-top:8px; margin-bottom:14px; }
.result-card .field-title { color:#2563eb; font-weight:700; font-size:13px;
                             margin-top:14px; margin-bottom:2px; }
.result-card .field-value { color:#111827; font-size:14px; white-space:pre-wrap; }
.result-card .meta-line { color:#4b5563; font-size:12px; margin-top:8px; }
.user-bubble { background:#2563eb; color:white; border-radius:12px; padding:10px 14px;
               margin-top:10px; display:inline-block; max-width:90%; white-space:pre-wrap; }
</style>
"""


def render_exchange(raw_input: str, result: PipelineResult, price):
    st.markdown(f'<div class="user-bubble">{html.escape(raw_input)}</div>', unsafe_allow_html=True)

    if result.error:
        st.error(result.error)
        return

    rules_txt = ", ".join(result.applied_rules) if result.applied_rules else "aucune"
    cost = estimate_cost_usd(result.total_tokens or 0, result.completion_tokens or 0, price)
    cost_txt = f"  •  coût ≈ ${cost:.6f}" if cost is not None else ""

    st.markdown(
        f"""
        <div class="result-card">
            <div class="field-title">message (Cleaning)</div>
            <div class="field-value">{html.escape(result.message) or "(vide)"}</div>
            <div class="field-title">metadata</div>
            <div class="field-value">{html.escape(result.metadata) or "(vide)"}</div>
            <div class="field-title">humanized_message</div>
            <div class="field-value">{html.escape(result.humanized_message) or "(vide)"}</div>
            <div class="meta-line">règles appliquées : {html.escape(rules_txt)}  •  latence : {result.latency_ms:.0f} ms{cost_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    final_json = json.dumps(
        {
            "message": result.message,
            "metadata": result.metadata,
            "humanized_message": result.humanized_message,
        },
        ensure_ascii=False,
        indent=2,
    )
    st.markdown("**sortie JSON finale**")
    st.code(final_json, language="json")


with tab_single:
    st.markdown(RESULT_CARD_CSS, unsafe_allow_html=True)
    st.caption(
        "Collez un « Input brut » (une entrée de la colonne du Google Sheets) et validez. "
        "Le pipeline appelle le modèle choisi pour le Cleaning, puis applique l'Humanizer en local (sans appel LLM). "
        "L'historique de la conversation reste affiché tant que vous ne rechargez pas la page ou ne cliquez pas sur « Effacer »."
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = _load_conversation()  # list of (raw_input, PipelineResult)

    top_col1, top_col2 = st.columns([1, 5])
    with top_col1:
        if st.button("🗑 Effacer la conversation"):
            st.session_state.conversation = []
            _save_conversation(st.session_state.conversation)
            st.rerun()
    with top_col2:
        if st.session_state.conversation:
            history_json = json.dumps(
                [
                    {
                        "input_raw": raw,
                        "message": r.message,
                        "metadata": r.metadata,
                        "humanized_message": r.humanized_message,
                        "applied_rules": r.applied_rules,
                        "error": r.error,
                    }
                    for raw, r in st.session_state.conversation
                ],
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                "💾 Télécharger l'historique (JSON)",
                data=history_json,
                file_name="conversation_humanizer.json",
                mime="application/json",
            )

    # Replay every past exchange so nothing disappears on the next message
    price = fetch_model_pricing(API_KEY).get(model) if st.session_state.conversation else None
    for raw, result in st.session_state.conversation:
        render_exchange(raw, result, price)

    with st.form(key="send_form", clear_on_submit=True):
        raw_input = st.text_area("Input brut", height=100, placeholder="Collez le message brut ici…")
        send = st.form_submit_button("Envoyer ➤", type="primary")

    if send and raw_input.strip():
        with st.spinner("Traitement en cours…"):
            pipeline = build_pipeline()
            try:
                result: PipelineResult = pipeline.run(raw_input)
            except Exception as e:
                result = PipelineResult(raw_input, "", "", "", [], 0.0, None, None, None, str(e))
        st.session_state.conversation.append((raw_input, result))
        _save_conversation(st.session_state.conversation)
        st.rerun()

# -- Batch tab ----------------------------------------------------------

with tab_batch:
    st.caption("Importez un fichier CSV/Excel contenant une colonne « Input brut » (ou équivalent).")
    uploaded = st.file_uploader("Fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded)
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding="latin-1")
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            df = None

        if df is not None:
            norm_map = {_normalize_col_name(c): c for c in df.columns}
            col = None
            for candidate in INPUT_COLUMN_CANDIDATES:
                if candidate in norm_map:
                    col = norm_map[candidate]
                    break

            if col is None:
                st.error(f"Aucune colonne 'Input brut' trouvée. Colonnes disponibles : {list(df.columns)}")
            else:
                st.success(f"{len(df)} lignes détectées, colonne « {col} » utilisée comme input.")
                if st.button("▶ Lancer le traitement du lot"):
                    pipeline = build_pipeline()
                    rows = df[col].fillna("").astype(str).tolist()
                    progress = st.progress(0.0)
                    status = st.empty()
                    results = []
                    for i, raw in enumerate(rows):
                        try:
                            res = pipeline.run(raw)
                        except Exception as e:
                            res = PipelineResult(raw, "", "", "", [], 0.0, None, None, None, str(e))
                        results.append(res)
                        progress.progress((i + 1) / len(rows))
                        status.text(f"{i + 1} / {len(rows)} traités…")

                    price = fetch_model_pricing(API_KEY).get(model)
                    out_rows = []
                    total_cost = 0.0
                    have_cost = False
                    for r in results:
                        d = r.to_dict()
                        d["applied_rules"] = ", ".join(r.applied_rules)
                        cost = estimate_cost_usd(r.total_tokens or 0, r.completion_tokens or 0, price)
                        d["cost_usd"] = cost
                        if cost is not None:
                            total_cost += cost
                            have_cost = True
                        out_rows.append(d)

                    out_df = pd.DataFrame(out_rows)
                    st.session_state["batch_results_df"] = out_df

                    failures = sum(1 for r in results if r.error)
                    st.write(f"Terminé : {len(results)} entrées, {failures} échec(s).")
                    if have_cost:
                        st.write(f"Coût total estimé : ${total_cost:.4f}")

                    st.dataframe(out_df, use_container_width=True)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("💾 Exporter les résultats (CSV)", data=csv_bytes,
                                        file_name="resultats_humanizer.csv", mime="text/csv")
