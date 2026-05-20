import os
import re
import time
import string
import pickle
import logging
import fitz
import pytesseract
from PIL import Image
from groq import Groq
from flask import send_from_directory

try:
    from google import genai
    from google.genai import types as genai_types
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logging.warning("google-genai not installed - Google fallback disabled.")

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Get a free key at: https://console.groq.com/keys"
    )

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH  = "legal_db"
BM25_INDEX_PATH = "legal_db/bm25_index.pkl"

PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FAST_MODEL     = "llama-3.1-8b-instant"
FALLBACK_MODEL = "gemini-2.0-flash"

RETRIEVAL_K         = 7
FINAL_TOP_K         = 10
JUDGMENT_BOOST      = 0.015
MCQ_ROUNDS_REQUIRED = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

groq_client   = Groq(api_key=GROQ_API_KEY)
gemini_client = None
if _GOOGLE_AVAILABLE and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logging.info("Google Gemini fallback client initialised.")

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
db             = None
chat_history   = []
bm25_index     = None
bm25_documents = []

diagnostic_state = {
    "active":         False,
    "rounds_done":    0,
    "original_query": "",
    "collected_info": [],
    "domains":        [],
    "language":       "en",   # detected language stored here for MCQ continuity
}

# ---------------------------------------------------------------------------
# EMERGENCY HELPLINES (India)
# ---------------------------------------------------------------------------
HELPLINES = {
    "national_emergency": "112",
    "women_helpline":     "1091",
    "child_helpline":     "1098",
    "police":             "100",
    "legal_aid":          "15100",
    "domestic_violence":  "181",
    "senior_citizen":     "14567",
}

# ---------------------------------------------------------------------------
# SUPPORTED LANGUAGES & SCRIPT DETECTION
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "en":  ("English",             "English"),
    "hi":  ("Hindi",               "Hindi (हिन्दी)"),
    "bn":  ("Bengali",             "Bengali (বাংলা)"),
    "te":  ("Telugu",              "Telugu (తెలుగు)"),
    "mr":  ("Marathi",             "Marathi (मराठी)"),
    "ta":  ("Tamil",               "Tamil (தமிழ்)"),
    "ur":  ("Urdu",                "Urdu (اردو)"),
    "gu":  ("Gujarati",            "Gujarati (ગુજરાતી)"),
    "kn":  ("Kannada",             "Kannada (ಕನ್ನಡ)"),
    "pa":  ("Punjabi",             "Punjabi (ਪੰਜਾਬੀ)"),
}

SCRIPT_PATTERNS = [
    (r"[\u0C80-\u0CFF]", "kn"),   # Kannada
    (r"[\u0B80-\u0BFF]", "ta"),   # Tamil
    (r"[\u0C00-\u0C7F]", "te"),   # Telugu
    (r"[\u0A80-\u0AFF]", "gu"),   # Gujarati
    (r"[\u0A00-\u0A7F]", "pa"),   # Gurmukhi (Punjabi)
    (r"[\u0980-\u09FF]", "bn"),   # Bengali
    (r"[\u0600-\u06FF]", "ur"),   # Urdu
    (r"[\u0900-\u097F]", "devanagari"),
]

_MARATHI_MARKERS = [
    "आहे", "नाही", "मला", "माझे", "माझा", "तुम्ही", "आपण",
    "करायचे", "सांगा", "झाले", "होते", "केले", "असेल",
]


def detect_language(text: str, frontend_lang: str = "en") -> str:
    if not text or not text.strip():
        return frontend_lang

    # Step 1: Script range detection
    for pattern, lang_code in SCRIPT_PATTERNS:
        if re.search(pattern, text):
            if lang_code != "devanagari":
                logging.info(f"Language detected via script: {lang_code}")
                return lang_code

            #Step 2: Devanagari disambiguation (Hindi vs Marathi)
            if any(marker in text for marker in _MARATHI_MARKERS):
                logging.info("Language detected: mr (Marathi markers found)")
                return "mr"

            # LLM disambiguation for ambiguous Devanagari
            prompt = (
                f'This text is written in Devanagari script: "{text[:200]}"\n'
                f"Is it Hindi or Marathi? Reply with only one word: Hindi or Marathi."
            )
            try:
                result = llm_generate(prompt, model=FAST_MODEL, max_tokens=5, temperature=0.0)
                if "marathi" in result.lower():
                    logging.info("Language detected: mr (LLM Devanagari disambiguation)")
                    return "mr"
                logging.info("Language detected: hi (LLM Devanagari disambiguation)")
                return "hi"
            except Exception:
                # Default Devanagari to Hindi
                return "hi"

    # Step 3: Latin script — check if it might be non-English 
    # Count words that look like they could be Hinglish or another Latin-script language
    # If the frontend already set a non-English language, trust it
    if frontend_lang in SUPPORTED_LANGUAGES and frontend_lang != "en":
        return frontend_lang

    # Default to English
    return "en"


def language_instruction(lang_code: str) -> str:
    """Return the LLM instruction string for the given language code."""
    info = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])
    return info[1]  # e.g. "Kannada (ಕನ್ನಡ)"


# ---------------------------------------------------------------------------
# INTENT TAXONOMY  — 14 intents across 4 categories
# ---------------------------------------------------------------------------
# Category A: Emergency & Procedural
INTENT_URGENT_HELP       = "URGENT_HELP"
INTENT_POLICE_INTERACTION= "POLICE_INTERACTION"
INTENT_LEGAL_PROCEDURE   = "LEGAL_PROCEDURE"

# Category B: Substantive Law & Strategy
INTENT_LEGAL_EXPLANATION = "LEGAL_EXPLANATION"
INTENT_LAW_COMPARISON    = "LAW_COMPARISON"
INTENT_PENALTY_INFO      = "PENALTY_INFO"
INTENT_CASE_SUMMARY      = "CASE_SUMMARY"
INTENT_LEGAL_STRATEGY    = "LEGAL_STRATEGY"

# Category C: Civil, Rights & Domain
INTENT_RIGHTS_CHECK      = "RIGHTS_CHECK"
INTENT_EVIDENCE_GUIDANCE = "EVIDENCE_GUIDANCE"
INTENT_JURISDICTION_CHECK= "JURISDICTION_CHECK"
INTENT_LIMITATION_CHECK  = "LIMITATION_CHECK"

# Category D: Document Operations
INTENT_DOC_DRAFTING      = "DOCUMENT_DRAFTING"
INTENT_DOC_ANALYSE       = "DOC_ANALYSE"
INTENT_DOC_VERIFY        = "DOC_VERIFY"

ALL_INTENTS = [
    INTENT_URGENT_HELP, INTENT_POLICE_INTERACTION, INTENT_LEGAL_PROCEDURE,
    INTENT_LEGAL_EXPLANATION, INTENT_LAW_COMPARISON, INTENT_PENALTY_INFO,
    INTENT_CASE_SUMMARY, INTENT_LEGAL_STRATEGY,
    INTENT_RIGHTS_CHECK, INTENT_EVIDENCE_GUIDANCE,
    INTENT_JURISDICTION_CHECK, INTENT_LIMITATION_CHECK,
    INTENT_DOC_DRAFTING, INTENT_DOC_ANALYSE, INTENT_DOC_VERIFY,
]

# Risk keyword triggers (checked BEFORE LLM classification for speed)
RISK_HIGH_KEYWORDS = [
    "beating", "hit me", "assault", "rape", "molest", "attack",
    "arrested", "detained", "locked up", "kidnap", "abduct",
    "domestic violence", "husband beating", "locked out",
    "evicted", "thrown out", "absconding", "threatening to kill",
    "just happened", "right now", "tonight", "emergency",
    "bleeding", "injured", "hurt", "danger", "scared",
]
RISK_MEDIUM_KEYWORDS = [
    "harassment", "encroachment", "cheating", "fraud", "stolen",
    "not paying salary", "not paying", "illegal", "forcibly",
    "blackmail", "extortion", "threatening",
]
# Urgency override words — always bump to URGENT_HELP
URGENCY_OVERRIDE_WORDS = [
    "right now", "just now", "just happened", "happening now",
    "tonight", "today", "emergency", "abhi", "abhi abhi",
    "kal raat", "please help", "help me", "kya karu",
]

# Hinglish / regional legal term normalization
HINGLISH_MAP = {
    "challan":     "traffic fine or charge sheet",
    "bainama":     "sale deed",
    "patwari":     "revenue officer / land records official",
    "tehsildar":   "sub-divisional revenue officer",
    "namantaran":  "mutation of land records",
    "zameen":      "land / property",
    "kiraya":      "rent",
    "makaan malik":"landlord",
    "kiraaydaar":  "tenant",
    "maamla":      "legal matter / case",
    "darkhast":    "application / petition",
    "adalat":      "court",
    "vakeel":      "lawyer / advocate",
    "muavza":      "compensation",
    "bail":        "bail",
    "FIR":         "First Information Report",
    "NCR":         "Non-Cognizable Report",
    "DCP":         "Deputy Commissioner of Police",
    "SP":          "Superintendent of Police",
}

# ---------------------------------------------------------------------------
# DOMAIN TAXONOMY
# ---------------------------------------------------------------------------
DOMAIN_MAP = {
    "CRIMINAL": [
        "fir", "arrest", "bail", "bns", "bnss", "bsa", "murder", "theft",
        "rape", "assault", "cognizable", "non-cognizable", "chargesheet", "custody",
        "police", "remand", "anticipatory bail", "dowry", "cruelty",
        "evidence", "witness", "acquittal", "conviction", "sentence",
        "ndps", "narcotic", "drug", "pocso", "child abuse", "juvenile",
        "ipc", "crpc", "bharatiya nyaya sanhita", "bharatiya nagarik",
    ],
    "FAMILY": [
        "divorce", "marriage", "alimony", "maintenance", "custody", "child",
        "adoption", "succession", "inheritance", "will", "hindu marriage",
        "special marriage", "guardianship", "domestic violence", "dowry",
        "matrimonial", "spouse", "separation", "mutual consent", "talaq",
        "hindu succession", "muslim personal law", "christian marriage",
    ],
    "PROPERTY": [
        "property", "land", "flat", "house", "rent", "lease", "tenant",
        "landlord", "registration", "stamp duty", "sale deed", "transfer",
        "easement", "encroachment", "mutation", "rera", "builder", "possession",
        "conveyance", "title", "mortgage", "plot", "agricultural land",
        "benami", "adverse possession", "zameen", "bainama", "patwari",
    ],
    "COMMERCIAL": [
        "contract", "agreement", "breach", "company", "partnership", "llp",
        "gst", "tax", "invoice", "cheque bounce", "negotiable instrument",
        "sale of goods", "arbitration", "insolvency", "bankruptcy", "ibc",
        "winding up", "shareholder", "director", "nclt", "debt", "recovery",
        "drt", "loan", "npa", "guarantee", "indemnity", "saas", "vendor",
    ],
    "LABOUR": [
        "salary", "wage", "payment", "employment", "job", "termination",
        "retrenchment", "layoff", "pf", "epf", "gratuity", "maternity",
        "leave", "industrial dispute", "strike", "lockout", "workman",
        "labour court", "esi", "provident fund", "minimum wage", "bonus",
        "sexual harassment", "posh", "workplace", "overtime",
    ],
    "CONSUMER": [
        "consumer", "product", "defect", "deficiency", "service", "refund",
        "compensation", "forum", "cdrc", "district commission", "insurance",
        "claim", "hospital", "medical negligence", "builder delay",
        "e-commerce", "online fraud", "mis-selling",
    ],
    "CONSTITUTIONAL": [
        "fundamental right", "article", "constitution", "writ", "habeas corpus",
        "mandamus", "certiorari", "prohibition", "quo warranto", "pil",
        "supreme court", "high court", "amendment", "basic structure",
        "equality", "freedom", "right to life", "directive principles",
        "reservation", "rti", "right to information",
    ],
    "DIGITAL_CYBER": [
        "cyber", "hacking", "data", "privacy", "online", "internet", "it act",
        "social media", "defamation online", "phishing", "fraud online",
        "dpdp", "data protection", "intermediary", "platform",
        "email", "computer", "digital signature", "e-contract", "whatsapp",
        "electronic evidence", "electronic record",
    ],
    "TAX": [
        "income tax", "itr", "return", "tds", "tax deduction", "assessment",
        "notice", "scrutiny", "refund", "capital gains", "gst", "vat",
        "service tax", "tax evasion", "appeal", "tribunal", "itat",
        "black money", "foreign income", "nri tax",
    ],
    "REAL_ESTATE": [
        "rera", "builder", "flat", "apartment", "delay", "possession",
        "carpet area", "super built up", "allotment", "registry",
        "home loan", "developer", "project", "completion certificate",
        "occupancy certificate", "maintenance charges",
    ],
    "MOTOR_ACCIDENT": [
        "accident", "vehicle", "motor", "mact", "compensation", "injury",
        "death", "hit and run", "insurance", "third party", "driving licence",
        "rash driving", "motor vehicles act", "solatium",
    ],
    "ANTI_CORRUPTION": [
        "bribe", "corruption", "public servant", "pca", "prevention of corruption",
        "lokpal", "lokayukta", "rti", "vigilance", "disproportionate assets",
        "misconduct", "government officer",
    ],
    "SC_ST": [
        "sc", "st", "scheduled caste", "scheduled tribe", "atrocities",
        "discrimination", "untouchability", "reservation", "dalit",
        "tribal", "poa act", "prevention of atrocities",
    ],
    "ENVIRONMENT": [
        "environment", "pollution", "ngt", "green tribunal", "forest",
        "wildlife", "hazardous", "industrial waste", "water", "air",
        "noise", "solid waste", "eia", "clearance",
    ],
    "ARBITRATION_ADR": [
        "arbitration", "mediation", "conciliation", "adr", "award",
        "arbitrator", "tribunal", "seat", "enforcement", "new york convention",
        "domestic arbitration", "international arbitration",
    ],
}

LANDMARK_JUDGMENTS = {
    "kesavananda", "maneka", "puttaswamy", "vishaka", "navtej", "johar",
    "arnesh kumar", "lalita kumari", "dk basu", "satender antil", "selvi",
    "vineeta sharma", "shayara bano", "sarla mudgal", "danial latifi",
    "suraj lamp", "indore development", "vidya drolia", "swiss ribbons",
    "innoventive", "shreya singhal", "olga tellis", "hussainara",
    "mc mehta", "oleum gas", "bandhua mukti", "parmanand katara",
}

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
def load_embedding_model():
    logging.info("Loading embedding model...")
    try:
        ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        return ef
    except Exception as e:
        logging.error(f"Embedding model error: {e}")
        return None

def load_vector_database(embedding_function):
    if not os.path.exists(VECTOR_DB_PATH):
        logging.error(f"Vector DB not found at {VECTOR_DB_PATH}")
        return None
    try:
        vdb = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_function,
        )
        logging.info("Vector DB loaded.")
        return vdb
    except Exception as e:
        logging.error(f"Vector DB load error: {e}")
        return None

def load_bm25_index():
    global bm25_index, bm25_documents
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pkl_path   = os.path.join(script_dir, BM25_INDEX_PATH)

    if os.path.exists(pkl_path):
        logging.info(f"Loading pre-built BM25 index from {pkl_path} ...")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            bm25_index = data["bm25"]
            bm25_documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data["chunks"], data["metas"])
            ]
            logging.info(f"BM25 index loaded: {len(bm25_documents):,} chunks.")
            return True
        except Exception as e:
            logging.error(f"Failed to load BM25 pkl: {e}. Will rebuild from ChromaDB.")
    return _rebuild_bm25_from_db()


def _rebuild_bm25_from_db():
    global bm25_index, bm25_documents
    if db is None:
        logging.error("ChromaDB not loaded — cannot build BM25 index.")
        return False
    try:
        FETCH_BATCH = 500
        all_ids = db.get(include=[])["ids"]
        total   = len(all_ids)
        logging.info(f"Fetching {total:,} chunks for BM25 rebuild...")
        corpus, docs = [], []
        for i in range(0, total, FETCH_BATCH):
            batch = db.get(ids=all_ids[i : i + FETCH_BATCH], include=["documents", "metadatas"])
            for text, meta in zip(batch["documents"], batch["metadatas"]):
                if text:
                    docs.append(Document(page_content=text, metadata=meta or {}))
                    tokens = (
                        text.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                    )
                    corpus.append(tokens)
        if corpus:
            bm25_index     = BM25Okapi(corpus)
            bm25_documents = docs
            logging.info(f"BM25 index rebuilt: {len(docs):,} chunks.")
            return True
        logging.error("BM25 rebuild produced no corpus.")
        return False
    except Exception as e:
        logging.error(f"BM25 rebuild error: {e}")
        return False

# ---------------------------------------------------------------------------
# PREPROCESSING — Hinglish / regional term normalization
# ---------------------------------------------------------------------------
def preprocess_query(query: str) -> tuple[str, list[str]]:
    """
    Normalize Hinglish and regional legal terms to English equivalents.
    Returns (normalized_query, list_of_detected_local_terms).
    Also detects Devanagari/regional script and flags for translation.
    """
    detected_terms = []
    normalized = query

    # Detect Devanagari script presence
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', query))
    if has_devanagari:
        logging.info("Devanagari script detected in query.")
        normalized = f"[Contains Hindi/regional text] {query}"

    # Normalize known Hinglish legal terms
    query_lower = query.lower()
    for term, meaning in HINGLISH_MAP.items():
        if term.lower() in query_lower:
            detected_terms.append(f"{term} ({meaning})")
            # Append meaning but keep the original term too
            if term.lower() not in normalized.lower() or meaning.lower() not in normalized.lower():
                normalized = normalized + f" [{term}: {meaning}]"

    if detected_terms:
        logging.info(f"Hinglish terms detected: {detected_terms}")

    return normalized.strip(), detected_terms


# ---------------------------------------------------------------------------
# RISK SCORING ENGINE
# ---------------------------------------------------------------------------
def score_risk(query: str) -> str:
    query_lower = query.lower()

    # Urgency override — always HIGH
    if any(w in query_lower for w in URGENCY_OVERRIDE_WORDS):
        return "HIGH"

    if any(w in query_lower for w in RISK_HIGH_KEYWORDS):
        return "HIGH"

    if any(w in query_lower for w in RISK_MEDIUM_KEYWORDS):
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# INTENT & CONTEXT DETECTION
# ---------------------------------------------------------------------------
def detect_intent_and_risk(query: str, risk_level: str) -> dict:
    # HIGH risk: force intent, skip LLM call for speed
    if risk_level == "HIGH":
        return {
            "primary_intent":    INTENT_URGENT_HELP,
            "secondary_intent":  None,
            "entities":          {},
            "jurisdiction_hint": None,
        }

    intent_list = "\n".join(f"- {i}" for i in ALL_INTENTS)

    entity_hint = (
        "Also identify (if present): Actor (who is doing what), "
        "Location/State (important for state-specific laws), "
        "Urgency words (e.g. 'now', 'tonight')."
    )

    prompt = f"""You are classifying an Indian legal query. Return a JSON object only — no prose.

Available primary intents:
{intent_list}

Definitions:
- URGENT_HELP: Active crisis happening now (assault, arrest, eviction)
- POLICE_INTERACTION: Issues with police (refusing FIR, illegal detention)
- LEGAL_PROCEDURE: How to file / step-by-step process
- LEGAL_EXPLANATION: What does a law/section mean (map old IPC to new BNS)
- LAW_COMPARISON: Compare two laws, bail types, courts
- PENALTY_INFO: What is the punishment for X
- CASE_SUMMARY: Explain a specific court judgment
- LEGAL_STRATEGY: Should I settle, which route is better
- RIGHTS_CHECK: What rights do I have as tenant/employee/etc.
- EVIDENCE_GUIDANCE: Is X admissible, how to preserve evidence
- JURISDICTION_CHECK: Where to file, which court
- LIMITATION_CHECK: Is it too late to file, what is the deadline
- DOCUMENT_DRAFTING: Draft a legal notice, agreement, complaint
- DOC_ANALYSE: What does this uploaded document say
- DOC_VERIFY: Is this document legally valid/compliant

{entity_hint}

Query: "{query}"

Respond with ONLY this JSON (no markdown):
{{
  "primary_intent": "<ONE intent from list>",
  "secondary_intent": "<ONE intent or null>",
  "actor": "<e.g. landlord, employer, police, husband — or null>",
  "location": "<state or city if mentioned — or null>",
  "urgency": "<high | medium | low>"
}}"""

    result = llm_generate(prompt, model=FAST_MODEL, max_tokens=120, temperature=0.05)

    # Parse JSON safely
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        import json
        parsed = json.loads(clean)

        primary = parsed.get("primary_intent", INTENT_RIGHTS_CHECK).upper()
        if primary not in ALL_INTENTS:
            primary = INTENT_RIGHTS_CHECK

        secondary = parsed.get("secondary_intent")
        if secondary:
            secondary = secondary.upper()
            if secondary not in ALL_INTENTS or secondary == primary:
                secondary = None

        entities = {
            "actor":    parsed.get("actor"),
            "location": parsed.get("location"),
            "urgency":  parsed.get("urgency", "low"),
        }

        logging.info(f"Intent: {primary} | Secondary: {secondary} | Entities: {entities}")
        return {
            "primary_intent":    primary,
            "secondary_intent":  secondary,
            "entities":          entities,
            "jurisdiction_hint": parsed.get("location"),
        }

    except Exception as e:
        logging.error(f"Intent JSON parse error: {e}. Raw: {result[:200]}")
        return {
            "primary_intent":    INTENT_RIGHTS_CHECK,
            "secondary_intent":  None,
            "entities":          {},
            "jurisdiction_hint": None,
        }


# ---------------------------------------------------------------------------
# DOMAIN DETECTION
# ---------------------------------------------------------------------------
def detect_domains_keyword(text: str) -> list:
    text_lower = text.lower()
    matched = [d for d, kws in DOMAIN_MAP.items() if any(kw in text_lower for kw in kws)]
    return matched if matched else ["GENERAL"]


def detect_domains(text: str) -> list:
    domain_list = "\n".join(f"- {d}" for d in DOMAIN_MAP.keys())
    prompt = f"""You are a legal domain classifier for Indian law.
Classify the query into one or more domains from the list.

Available domains:
{domain_list}

Rules:
- Return only domain names, comma-separated
- No explanations
- If truly general, return: GENERAL
- Maximum 3 domains

Query: "{text}"

Domains:"""

    try:
        result = llm_generate(prompt, model=FAST_MODEL, max_tokens=50, temperature=0.05)
        if not result or result.startswith("Service"):
            return detect_domains_keyword(text)
        domains = [d.strip().upper().replace(" ", "_") for d in result.split(",")]
        valid   = [d for d in domains if d in DOMAIN_MAP or d == "GENERAL"]
        if valid:
            logging.info(f"LLM domain classification: {valid}")
            return valid
    except Exception as e:
        logging.error(f"LLM domain detection error: {e}")
    return detect_domains_keyword(text)


def is_landmark(doc: Document) -> bool:
    source  = doc.metadata.get("source", "").lower()
    content = doc.page_content.lower()
    return any(lm in source or lm in content for lm in LANDMARK_JUDGMENTS)


# ---------------------------------------------------------------------------
# CENTRAL LLM CALL
# ---------------------------------------------------------------------------
def initialize_ai():
    global db

    if db is None:
        logging.info("Initializing AI components...")

        ef = load_embedding_model()

        if ef:
            db = load_vector_database(ef)

        if db and bm25_index is None:
            load_bm25_index()

def llm_generate(
    prompt: str,
    system: str = "",
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.25,
) -> str:
    target_model = model or PRIMARY_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    groq_retries = 2
    for attempt in range(groq_retries):
        try:
            resp = groq_client.chat.completions.create(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.90,
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip()
            logging.warning(f"Groq returned empty response (model={target_model})")
            break
        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "quota" in err:
                wait = 8 * (attempt + 1)
                logging.warning(f"Groq rate-limit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                continue
            if target_model == FAST_MODEL:
                logging.warning(f"FAST_MODEL error ({e}). Retrying with PRIMARY_MODEL...")
                target_model = PRIMARY_MODEL
                continue
            logging.error(f"Groq error: {e}")
            break

    if gemini_client:
        logging.warning("Falling back to Google Gemini...")
        try:
            full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
            g_resp = gemini_client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=min(temperature + 0.05, 0.5),
                    top_p=0.90,
                    max_output_tokens=max_tokens,
                ),
            )
            if g_resp and g_resp.text:
                return g_resp.text.strip()
        except Exception as ge:
            logging.error(f"Google fallback also failed: {ge}")

    return "Service temporarily unavailable. Please try again in a moment."


# ---------------------------------------------------------------------------
# QUERY TRANSFORMATION
# ---------------------------------------------------------------------------
def rephrase_query(query: str, language: str) -> str:
    if not chat_history:
        return query
    history_str = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    )
    prompt = (
        f"Rewrite the follow-up query as a complete standalone legal question in {language}.\n\n"
        f"Chat History:\n{history_str}\n\n"
        f"Follow-up: \"{query}\"\n\n"
        f"Standalone query (output only the rewritten query, nothing else):"
    )
    result = llm_generate(prompt, model=FAST_MODEL, max_tokens=200, temperature=0.15)
    logging.info(f"Rephrased query: {result}")
    return result if result and not result.startswith("Service") else query


def expand_query(query: str, language: str, domains: list) -> list:
    domain_hint = ", ".join(domains)
    prompt = (
        f"You are a legal search assistant working with Indian law (BNS/BNSS/BSA era). "
        f"The query relates to: {domain_hint}.\n"
        f"Write 4 different search queries to retrieve relevant legal content. "
        f"Include specific section numbers from BNS/BNSS/BSA where relevant, and landmark case names.\n"
        f"Language: {language}.\n\n"
        f"User Query: \"{query}\"\n\n"
        f"Write one search query per line, no numbering or bullets."
    )
    result = llm_generate(prompt, model=FAST_MODEL, max_tokens=300, temperature=0.2)
    if result.startswith("Service"):
        return [query]
    variants = [l.strip() for l in result.split("\n") if l.strip()][:4]
    variants.append(query)
    logging.info(f"Expanded queries: {variants}")
    return variants


# ---------------------------------------------------------------------------
# VAGUENESS DETECTION
# ---------------------------------------------------------------------------
def is_query_vague(query: str, risk_level: str) -> bool:
    # HIGH risk queries are never vague — act immediately
    if risk_level == "HIGH":
        return False

    prompt = f"""You are reviewing a user's legal question.

A question needs more information if:
- It is missing key facts (who, what happened, which state)
- It is under 10 words with no specific legal issue

A question can be answered directly if:
- It asks about a specific law, section, or judgment
- It has enough facts to give a clear legal answer
- It is a "what is" or "how does" type question
- It describes a problem clearly even without all details

Question: "{query}"

Reply with only one word: VAGUE or SPECIFIC"""

    result = llm_generate(prompt, model=FAST_MODEL, max_tokens=10, temperature=0.05)
    is_vague = "VAGUE" in result.upper()
    logging.info(f"Vagueness check: '{query[:60]}' -> {result.strip()}")
    return is_vague


# ---------------------------------------------------------------------------
# MCQ DIAGNOSTIC GENERATION
# ---------------------------------------------------------------------------
def generate_mcq(
    original_query: str,
    domains: list,
    collected_info: list,
    language: str,
    round_num: int,
) -> str:
    collected_str = ""
    if collected_info:
        collected_str = "\n".join(
            f"  Round {i+1}: {info}" for i, info in enumerate(collected_info)
        )
    domain_str = ", ".join(domains)

    prompt = f"""You are a legal assistant helping someone in India with a legal problem.
Their question needs one focused follow-up before you can give accurate advice.

Original question: "{original_query}"
Legal area(s): {domain_str}
Question round: {round_num} of {MCQ_ROUNDS_REQUIRED}
What you already know:
{collected_str if collected_str else "  Nothing yet."}

Guidelines:
- Ask the single most important thing you still need to know
- Give exactly 3 or 4 answer choices (A, B, C or A, B, C, D)
- Keep the question short and clear for a non-lawyer
- Do not ask for personal identifying details
- Respond in: {language}

Format exactly like this:

**To give you the most accurate advice, I need one quick clarification:**

[Your question here]

**A)** [Option A]
**B)** [Option B]
**C)** [Option C]
**D)** [Option D — only if needed]

*You can also type your own answer if none of these options fit your situation.*"""

    return llm_generate(prompt, model=PRIMARY_MODEL, max_tokens=400, temperature=0.45)


def classify_mcq_answer(answer: str, original_query: str, domains: list) -> str:
    prompt = f"""Someone answered a follow-up question about their legal problem.
Original question: "{original_query}"
Their answer: "{answer}"

Summarise as a single factual sentence of no more than 20 words.
Output only the summary sentence."""

    return llm_generate(prompt, model=FAST_MODEL, max_tokens=60, temperature=0.1)


# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL
# ---------------------------------------------------------------------------
def hybrid_search(query: str, k: int = RETRIEVAL_K) -> list:
    global db, bm25_index, bm25_documents

    vector_docs = []
    try:
        vector_docs = db.similarity_search(query, k=k)
    except Exception as e:
        logging.error(f"Dense search error: {e}")

    bm25_docs = []
    if bm25_index:
        try:
            tokens = (
                query.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
            )
            bm25_docs = bm25_index.get_top_n(tokens, bm25_documents, n=k)
        except Exception as e:
            logging.error(f"BM25 search error: {e}")

    rrf_k  = 60
    scores = {}
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (rank + rrf_k)
        if is_landmark(doc):
            scores[key]["score"] += JUDGMENT_BOOST

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1.0 / (rank + rrf_k)
        if is_landmark(doc):
            scores[key]["score"] += JUDGMENT_BOOST

    reranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked[:k]]


def retrieve_context(query: str, language: str):
    if not db:
        return "Vector database not loaded.", [], []

    domains    = detect_domains(query)
    standalone = rephrase_query(query, language)
    variants   = expand_query(standalone, language, domains)

    seen = {}
    for v in variants:
        for doc in hybrid_search(v, k=RETRIEVAL_K):
            if doc.page_content not in seen:
                seen[doc.page_content] = doc

    top_docs  = sorted(seen.values(), key=lambda d: 1.0 if is_landmark(d) else 0.0, reverse=True)[:FINAL_TOP_K]
    statutes  = [d for d in top_docs if d.metadata.get("doc_type") != "JUDGMENT"]
    judgments = [d for d in top_docs if d.metadata.get("doc_type") == "JUDGMENT"]

    parts        = []
    source_files = set()

    if statutes:
        parts.append("### STATUTORY PROVISIONS ###\n")
        for doc in statutes:
            src = doc.metadata.get("source", "Unknown")
            source_files.add(src)
            parts.append(f"[Source: {src}]\n{doc.page_content}\n---\n")

    if judgments:
        parts.append("\n### LANDMARK JUDGMENTS ###\n")
        for doc in judgments:
            src = doc.metadata.get("source", "Unknown")
            source_files.add(src)
            parts.append(f"[Judgment: {src}]\n{doc.page_content}\n---\n")

    context = "\n".join(parts)
    logging.info(
        f"Retrieved {len(statutes)} statute + {len(judgments)} judgment chunks "
        f"from {len(source_files)} sources. Domains: {domains}"
    )
    return context, list(source_files), domains


# ---------------------------------------------------------------------------
# RESPONSE CLEANING
# ---------------------------------------------------------------------------
_BANNED = re.compile(
    r"(STEP\s*\d+:?\s*\w*|TRIAGE\s*\(.*?\)|DIAGNOSE\s*\(.*?\)|"
    r"The user'?s query is|This is a general[,\s]|I will proceed to|"
    r"Go to STEP|Move to approach|Analyze the query|SIMPLE ANSWER|DETAILED ANSWER)",
    re.IGNORECASE,
)

def clean_response(text: str) -> str:
    text = _BANNED.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# DOMAIN-SPECIFIC GUIDANCE STRINGS (updated for BNS/BNSS/BSA era)
# ---------------------------------------------------------------------------
DOMAIN_GUIDANCE = {
    "CRIMINAL": (
        "IMPORTANT: India replaced IPC with Bharatiya Nyaya Sanhita (BNS) 2023, "
        "CrPC with Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, and Indian Evidence Act "
        "with Bharatiya Sakshya Adhiniyam (BSA) 2023 — effective July 1, 2024. "
        "Always cite BNS/BNSS/BSA sections. When referencing old IPC sections, note the "
        "equivalent BNS section (e.g. 'formerly Section 420 IPC, now Section 318 BNS'). "
        "Cover rights during arrest under BNSS Section 35 (formerly CrPC 41A), "
        "bail under BNSS, FIR procedure, and DK Basu / Arnesh Kumar safeguards."
    ),
    "FAMILY": (
        "Identify which personal law applies — Hindu, Muslim, Christian, Parsi, or Special Marriage Act. "
        "Reference Vineeta Sharma (2020) for coparcenary rights, Shayara Bano (2017) for triple talaq. "
        "Maintenance under BNSS Section 144 (formerly CrPC 125). "
        "Cover Domestic Violence Act 2005 for protection orders and residence rights."
    ),
    "PROPERTY": (
        "Cover Transfer of Property Act 1882, Registration Act 1908, and RERA 2016 as applicable. "
        "Reference Suraj Lamp (2011) when GPA/power of attorney sales arise — they are not valid transfers. "
        "Address stamp duty (state-specific), registration requirements, and 12-year limitation for adverse possession. "
        "For tenant disputes, state Rent Control Acts vary — flag this."
    ),
    "COMMERCIAL": (
        "Reference Indian Contract Act 1872, NI Act Section 138 for cheque bounce (summary trial, 2-year limitation), "
        "and IBC 2016 backed by Swiss Ribbons (2019) and Innoventive Industries (2017). "
        "For cross-border: Arbitration and Conciliation Act 1996 (amended 2019/2021). "
        "NCLT for insolvency/company matters, DRT for debt recovery above ₹20 lakh."
    ),
    "LABOUR": (
        "Four Labour Codes (2019-2020) consolidate old laws but implementation is state-specific — "
        "many states still operate under old Acts. Reference as applicable: "
        "Code on Wages, EPF Act 1952, Gratuity Act 1972, Maternity Benefit Act 1961. "
        "POSH Act 2013 and Vishaka (1997) for sexual harassment at workplace. "
        "Industrial Disputes Act for retrenchment — 3-year limitation generally."
    ),
    "CONSUMER": (
        "Consumer Protection Act 2019. Three-tier jurisdiction: "
        "District Commission up to ₹50 lakh, State Commission ₹50 lakh to ₹2 crore, "
        "National Commission (NCDRC) above ₹2 crore. "
        "Two-year limitation from cause of action. E-commerce covered under 2020 Rules."
    ),
    "CONSTITUTIONAL": (
        "Cite relevant Articles and landmark cases. "
        "Kesavananda Bharati (1973) — basic structure doctrine. "
        "Puttaswamy (2017) — right to privacy as fundamental right. "
        "Maneka Gandhi (1978) — due process under Article 21. "
        "Writ jurisdiction: Article 226 (High Courts), Article 32 (Supreme Court)."
    ),
    "DIGITAL_CYBER": (
        "Reference IT Act 2000, DPDP Act 2023 (Digital Personal Data Protection), IT Rules 2021. "
        "Electronic records under BSA Section 63 (formerly IEA Section 65B) — "
        "certificate requirement for admissibility. "
        "Sections 66 (hacking), 66C (identity theft), 66D (impersonation), 67 (obscene content). "
        "WhatsApp messages are admissible with proper Section 63 BSA certificate."
    ),
    "TAX": (
        "Income Tax Act 1961, GST Act 2017. "
        "Appeal path: CIT(Appeals) → ITAT → High Court → Supreme Court. "
        "Cover notice response timelines and penalty provisions. "
        "Faceless assessment under Section 144B."
    ),
    "REAL_ESTATE": (
        "RERA Act 2016. Builder must register project before launch. "
        "Section 18: interest on delayed possession (SBI MCLR + 2%). "
        "RERA Authority, then Appellate Tribunal. "
        "Homebuyers are financial creditors under IBC — can approach NCLT."
    ),
    "MOTOR_ACCIDENT": (
        "Motor Vehicles Act 1988 (amended 2019). MACT jurisdiction. "
        "Sarla Verma (2009) structured formula for compensation. "
        "Hit-and-run: Solatium Fund (₹2 lakh for death, ₹50,000 for grievous hurt). "
        "No-fault liability under Section 163A."
    ),
    "ANTI_CORRUPTION": (
        "Prevention of Corruption Act 1988 (amended 2018). "
        "Lokpal (central), Lokayukta (state) jurisdiction. "
        "Whistle Blowers Protection Act 2014. "
        "Giving a bribe is now also an offence under 2018 amendment."
    ),
    "SC_ST": (
        "SC/ST (Prevention of Atrocities) Act 1989 and Amendment 2018. "
        "Special courts, anticipatory bail generally not available (Subhash Kashinath Mahajan, 2018 — note Supreme Court restored original stringency). "
        "Reservation rights under Articles 15, 16, 342 of the Constitution."
    ),
    "ENVIRONMENT": (
        "Environment Protection Act 1986, Water Act 1974, Air Act 1981, NGT Act 2010. "
        "NGT has original jurisdiction for civil cases with substantial environmental questions. "
        "M.C. Mehta judgments: polluter pays principle, absolute liability."
    ),
    "ARBITRATION_ADR": (
        "Arbitration and Conciliation Act 1996 (amended 2015, 2019, 2021). "
        "Vidya Drolia (2021) — arbitrability test. Section 8: referral to arbitration. "
        "Section 34: challenge to award (90 days + 30 day grace). "
        "Seat vs venue: seat determines curial law (BGS SGS Soma JV, 2019)."
    ),
}


# ---------------------------------------------------------------------------
# RESPONSE FORMAT TEMPLATES — one per intent
# ---------------------------------------------------------------------------
def _helpline_block() -> str:
    return (
        "\n**Emergency helplines:**\n"
        f"- National Emergency: **{HELPLINES['national_emergency']}**\n"
        f"- Women's Helpline: **{HELPLINES['women_helpline']}**\n"
        f"- Police: **{HELPLINES['police']}**\n"
        f"- Free Legal Aid: **{HELPLINES['legal_aid']}**\n"
    )


RESPONSE_FORMAT_TEMPLATES = {

    INTENT_URGENT_HELP: f"""
The user is in an active crisis. Respond like a calm, knowledgeable friend — not a textbook.
Lead with ACTION, not legal theory. Be warm and direct.

## What to do right now
1. **[Most urgent action]** — [one plain sentence explaining why]
2. **[Second action]** — [one sentence]
3. **[Third action]** — [one sentence]

{_helpline_block()}

## Your rights in this situation
- **[Right 1]** — plain language, one sentence
- **[Right 2]** — plain language, one sentence

## The legal basis
(Now bring in citations — keep this section short.)
**Section X, BNS/BNSS/BSA or relevant Act** [source: file.pdf] — what it covers.
**Case Name (Year)** — what it established.

## Do not miss these deadlines
- **[Deadline]:** [timeframe] — [consequence if missed]

## Practical tips
1. [Real-world tip — e.g. record everything on phone, tell a trusted person]
2. [Another concrete tip]

Tone: Urgent, warm, human. No legal preamble. Get straight to what they need to do.""",

    INTENT_POLICE_INTERACTION: f"""
The user has a problem with police (refusing FIR, illegal detention, etc.).
Be precise about their rights and the escalation path.

## Your rights when dealing with police
- **Right to FIR:** Under BNSS Section 173, police MUST register an FIR for cognizable offences.
  Zero FIR can be filed at any police station, not just the one with jurisdiction.
- **Right against illegal detention:** [add relevant BNSS section]
- **[Other right]:** [plain sentence]

## Step-by-step: what to do if police refuse
1. **Ask for written refusal** — [why this matters]
2. **E-FIR / online FIR** — available in most states for certain offences
3. **Written complaint to SP/DCP** — [how to send, via registered post]
4. **Magistrate complaint under BNSS Section 175** — [when to use this]
5. **High Court writ (Habeas Corpus / Mandamus)** — [last resort]

{_helpline_block()}

## Legal citations
**BNSS Section 173** [source: file.pdf] — FIR registration duty
**Lalita Kumari (2014)** — mandatory FIR for cognizable offences

## Practical tips
1. [Concrete tip — e.g. always get acknowledgement slip]
2. [Another tip]

Tone: Firm and empowering. The user should know exactly what leverage they have.""",

    INTENT_LEGAL_PROCEDURE: """
The user wants step-by-step instructions for a legal process.
Be precise, sequential, and practical. Include realistic timelines.

## How to [procedure name]: step-by-step

**Step 1 — [Action]**
[What to do, where to go, what document to bring. One short paragraph.]

**Step 2 — [Action]**
[Repeat format]

**Step 3 — [Action]**
[Repeat format]

## Required documents
- [Document 1]
- [Document 2]
- [Document 3]

## Where to file
- **Authority:** [exact name — e.g. District Consumer Commission, RERA Authority, Labour Court]
- **Fee:** [filing fee if known]
- **Realistic timeline:** [how long this actually takes in India]

## Legal basis
**Section X, Act Name** [source: file.pdf] — what it authorizes

## Common mistakes to avoid
1. [Mistake people make — and how to avoid it]
2. [Another]

Tone: Clear and practical. Like a guide written by someone who has done this before.""",

    INTENT_LEGAL_EXPLANATION: """
The user wants to understand a law or section. Teach it clearly.
IMPORTANT: Always map old IPC sections to new BNS equivalents where applicable.

## What [law/section] means

(2-3 sentences in plain language. No jargon. Explain to an intelligent non-lawyer.)

## The old vs. new law (if applicable)
| | Old Law | New Law |
|---|---|---|
| **Section** | IPC / CrPC / IEA | BNS / BNSS / BSA |
| **Number** | [old section] | [new section] |
| **Key change** | [if any] | [description] |

## Essential ingredients
For this law to apply, ALL of the following must be present:
1. [Element 1]
2. [Element 2]
3. [Element 3]

## Real-world example
[A plain scenario showing how this law works in practice — 3-4 sentences]

## Related laws worth knowing
[1-2 connected sections or acts]

## Full citation
**Section X, BNS/Act Name** [source: file.pdf]

Tone: Educational and conversational. Like a law professor explaining to a first-year student.""",

    INTENT_LAW_COMPARISON: """
The user wants to compare two laws, offences, or procedures.

## [Law A] vs [Law B]: at a glance

| Factor | [Law A] | [Law B] |
|---|---|---|
| Definition | | |
| Cognizable? | | |
| Bailable? | | |
| Court | | |
| Punishment | | |
| Who can file | | |

## Key difference in plain language
[2-3 sentences on the practical significance of the difference]

## Which one applies to you?
[Direct guidance on which scenario leads to which law]

## Legal citations
**[Law A]:** Section X, Act Name [source: file.pdf]
**[Law B]:** Section X, Act Name [source: file.pdf]

Tone: Direct and scannable. The table should do most of the work.""",

    INTENT_PENALTY_INFO: """
The user wants to know the punishment for an offence.

## Punishment for [offence]

Under **Section X, BNS** (formerly Section Y, IPC):

| | Details |
|---|---|
| **Imprisonment** | [minimum] to [maximum] |
| **Fine** | [amount or "as decided by court"] |
| **Cognizable?** | Yes / No |
| **Bailable?** | Yes / No |
| **Compoundable?** | Yes / No (meaning: can it be settled between parties?) |
| **Trial court** | [which court] |

## What "cognizable" means here
[One sentence — police can arrest without warrant, or cannot]

## Aggravating factors that increase punishment
- [Factor 1]
- [Factor 2 if applicable]

## Full citation
**Section X, BNS 2023** [source: file.pdf] (formerly IPC Section Y)

Tone: Factual and precise. No padding.""",

    INTENT_CASE_SUMMARY: """
The user wants to understand a court judgment.

## [Case Name] ([Year])
**Court:** [Supreme Court / High Court name]
**Bench:** [if notable — e.g. Constitution Bench]

## What the case was about
[2-3 sentences on the facts and the legal question the court had to answer.]

## What the court decided
[The actual ruling — be precise. What was upheld, struck down, or established?]

## The principle it established
[The ratio decidendi — the legal rule that came out of this case, in plain language]

## Why it still matters
[How courts and lawyers use this judgment today. What it changed.]

## One line that captures it
"[A paraphrase of the core principle in under 15 words]"

## Where it applies
[Situations or laws this judgment is directly relevant to]

Tone: Engaging and clear. Help the user feel why this case was significant.""",

    INTENT_LEGAL_STRATEGY: """
The user is deciding between legal options. Give an honest, balanced analysis.

## The core question
[Restate what they are deciding in one sentence.]

## Option A: [e.g. File criminal complaint / FIR]
**Pros:**
- [Advantage 1]
- [Advantage 2]

**Cons / Risks:**
- [Disadvantage 1]
- [Disadvantage 2]

**Realistic cost:** [filing fee, lawyer fee estimate]
**Realistic timeline:** [how long this route takes in India]

## Option B: [e.g. Civil suit / Mediation / Settlement]
**Pros:**
- [Advantage 1]

**Cons / Risks:**
- [Disadvantage 1]

**Realistic cost:** [estimate]
**Realistic timeline:** [estimate]

## What most lawyers would advise
[Your honest read on which option is stronger given the facts — be direct, not wishy-washy]

## One thing to do before deciding
[A concrete step — e.g. send legal notice, gather evidence, consult a specialist]

Tone: Like a trusted advisor, not a hedge-everything lawyer. Be direct.""",

    INTENT_RIGHTS_CHECK: """
The user wants to know their rights in a specific situation. Empower them.

## Your rights as [role/situation]

Each right stated clearly, with the law behind it.

- **Right to [X]**
  *What this means:* [plain language sentence]
  *Law:* Section X, Act Name

- **Right to [Y]**
  *What this means:* [plain language sentence]
  *Law:* Case Name (Year) / Section X, Act Name

- **Right to [Z]**
  *What this means:* [plain language sentence]
  *Law:* Section X, Act Name

## If your rights are being violated
1. **[First step]** — [one sentence]
2. **[Second step]** — [one sentence]
3. **[Escalation]** — [one sentence]

## What you cannot do
[Any important limits on these rights — keep brief, only if genuinely relevant]

## Full citations
- **Section X, Act Name** [source: file.pdf]
- **Case Name (Year)**

Tone: Empowering. The user should finish reading feeling informed and capable.""",

    INTENT_EVIDENCE_GUIDANCE: """
The user wants to know about evidence — admissibility, preservation, or strength.

## Is [evidence type] admissible in Indian courts?

**Short answer:** [Yes / No / Yes, with conditions]

## The legal rule
Under **BSA Section 63** (formerly IEA Section 65B) — electronic records including
WhatsApp messages, emails, screenshots, and CCTV footage are admissible IF:

1. **Certificate requirement:** A certificate from the person responsible for the device
   stating the record was produced by that computer/system. Without this, the evidence
   may be excluded.
2. **Integrity:** The record must not have been tampered with.
3. **Authenticity:** Must be linked to a specific device and account.

## How to preserve this evidence properly
1. [Preservation step 1 — e.g. take screenshots with timestamps visible]
2. [Step 2 — e.g. get CDR / call records via police]
3. [Step 3 — e.g. request platform data before it's deleted]

## Strength in court
**Strong evidence:** [types that courts weigh heavily]
**Weak without support:** [types that need corroboration]

## Citation
**BSA Section 63** [source: file.pdf] — electronic record admissibility
**Arjun Panditrao Khotkar (2020)** — certificate requirement for electronic records

Tone: Practical and precise. Tell them exactly what to do to preserve evidence.""",

    INTENT_JURISDICTION_CHECK: """
The user wants to know where to file their case.

## Where to file: the two tests

**Test 1 — Territorial jurisdiction** (where the case must go geographically)
[Which court has territorial power: where cause of action arose, where defendant lives, etc.]

**Test 2 — Pecuniary jurisdiction** (which court level based on the amount)
| Amount involved | Court |
|---|---|
| Up to ₹[X] | [Court name] |
| ₹[X] to ₹[Y] | [Court name] |
| Above ₹[Y] | [Court name] |

## The right authority for your situation
**File at:** [specific court / forum / authority with address if possible]
**Because:** [one sentence explaining why this is the correct forum]

## If jurisdiction is disputed
[One sentence on what happens if the other side challenges jurisdiction]

## Citation
**Section X, CPC / relevant Act** [source: file.pdf] — jurisdiction rules

Tone: Direct. Give them the specific answer, not a law school lecture.""",

    INTENT_LIMITATION_CHECK: """
The user wants to know if it's too late to file, or what their deadline is.

## The limitation period for your case

Under the **Limitation Act 1963**, the time limit for [type of case] is:

**[X years / months]** from the date [trigger event — e.g. cause of action arose / you became aware].

## Your deadline
[If they gave you a date: "Since [event] happened on [date], your deadline is approximately [calculated date]."
If no date: "You need to calculate from the date [trigger event]."]

## Exceptions that can save a late claim
1. **Condonation of delay (Section 5):** Courts can excuse delay if you show "sufficient cause."
   Common accepted reasons: illness, being out of the country, not knowing about the right.
2. **Fraud or concealment (Section 17):** Limitation runs from when you discovered the fraud.
3. **Minor or disability (Section 6):** Time starts when the disability ends.

## What happens if you miss the deadline
[Honest answer — case is barred, but explain if there is any remedy]

## Citation
**Limitation Act 1963, Article [X], Schedule** [source: file.pdf]
**[Relevant case on condonation]** — standard courts apply

Tone: Precise and honest. If it's too late, say so clearly — but also show every exception.""",

    INTENT_DOC_DRAFTING: """
The user wants a legal document drafted.

Provide a complete, usable draft with clear placeholders.

---

**[DOCUMENT TYPE]**

[City], [Date]

**To,**
[NAME OF RECIPIENT]
[DESIGNATION]
[ORGANIZATION]
[ADDRESS]

**Subject:** [SUBJECT IN BOLD]

Sir/Madam,

[Opening paragraph — state who you are and the relationship/transaction]

[Body paragraph 1 — state the facts chronologically]

[Body paragraph 2 — state what went wrong / the grievance]

[Body paragraph 3 — state the legal basis for your claim, citing the relevant section]

You are hereby called upon to [SPECIFIC DEMAND — e.g. pay ₹X within 15 days /
vacate the premises / restore my service].

Failing which, I shall be constrained to initiate appropriate legal proceedings
before the competent court/authority without further notice, at your risk as to cost.

Yours faithfully,
[YOUR FULL NAME]
[ADDRESS]
[CONTACT]

---

**How to use this draft:**
1. Replace every item in [SQUARE BRACKETS] with your actual details
2. Have it typed on plain paper (legal notice does not require stamp paper)
3. Send by Registered Post AD — keep the receipt and acknowledgement card

**Legal basis for this notice:** Section X, Act Name

Tone: Formal and precise. The draft must be usable as-is once placeholders are filled.""",

    INTENT_DOC_ANALYSE: """
The user has uploaded a document and wants to understand it.

## What this document is
[1-2 sentences identifying the document type, parties, and purpose.]

## Key provisions in plain language

**[Clause/Section name]:**
[What it means in plain English — one short paragraph]

**[Another clause]:**
[What it means]

**[Important clause]:**
[What it means]

(Cover the 5-6 most important provisions.)

## What you are agreeing to / what this means for you
[2-3 sentences on the practical implications — rights, obligations, risks]

## Anything unusual or one-sided
(Flag any clause that is non-standard, unusually restrictive, or that deserves attention.
Be direct — don't soften important warnings.)
- **[Clause]:** [Why it's unusual and what it means]

## Applicable law
**[Relevant Act]** governs this type of document — [one sentence on what that means practically]

Tone: Plain English throughout. The user should understand their document without needing a lawyer.""",

    INTENT_DOC_VERIFY: """
The user wants to know if their document is legally valid or compliant.

## Overall assessment
[One sentence verdict: broadly compliant / has significant issues / invalid on the face of it]

## What is correctly included
- [Compliant element 1]
- [Compliant element 2]
- [Compliant element 3]

## Issues and red flags
(Be direct. Don't minimize serious problems.)

- **[Issue 1]:** [What is wrong, why it matters legally, and what risk it creates]
- **[Issue 2]:** [Same format]
- **[Missing element]:** [What should be there but isn't]

## What the law requires
**Section X, Act Name** [source: file.pdf] — [what this section mandates for this document type]
**Section Y, Act Name** [source: file.pdf] — [another requirement]

## Recommended next steps
1. **[Fix this first]:** [specific action]
2. **[Then do this]:** [specific action]
3. **Get a lawyer to review if:** [specific conditions that make professional review necessary]

Tone: Direct and honest. A serious problem gets a serious warning. A minor gap gets a minor note.""",
}


# ---------------------------------------------------------------------------
# BUILD SYSTEM PROMPT — intent-aware
# ---------------------------------------------------------------------------
def build_system_prompt(
    domains: list,
    language: str,
    enriched_context: str = "",
    intent: str = INTENT_URGENT_HELP,
    risk_level: str = "LOW",
    entities: dict = None,
    secondary_intent: str = None,
    local_terms: list = None,
) -> str:

    entities = entities or {}

    # Active domain guidance
    active_guidance = "\n".join(
        f"  - {DOMAIN_GUIDANCE[d]}" for d in domains if d in DOMAIN_GUIDANCE
    ) or "  - Apply general Indian legal principles carefully."

    # Entity context block
    entity_block = ""
    if entities.get("actor"):
        entity_block += f"\nActor identified in query: {entities['actor']}"
    if entities.get("location"):
        entity_block += f"\nLocation/State mentioned: {entities['location']} — apply state-specific laws where relevant."
    if local_terms:
        entity_block += f"\nLocal/Hinglish terms detected: {', '.join(local_terms)}"

    enriched = f"\nContext about the user's situation:\n{enriched_context}\n" if enriched_context else ""

    # Secondary intent addendum
    secondary_block = ""
    if secondary_intent and secondary_intent in RESPONSE_FORMAT_TEMPLATES:
        secondary_block = (
            f"\n\nSECONDARY INTENT ({secondary_intent}) — after addressing the primary response, "
            f"briefly address this secondary need as well, using its relevant format elements."
        )

    # Risk banner for HIGH risk
    risk_banner = ""
    if risk_level == "HIGH":
        risk_banner = (
            "\n HIGH RISK QUERY — The user may be in immediate danger or crisis.\n"
            "Suppress long legal explanations. Lead with safety and immediate action.\n"
            "Include emergency helplines prominently.\n"
        )

    # Primary format template
    format_template = RESPONSE_FORMAT_TEMPLATES.get(
        intent, RESPONSE_FORMAT_TEMPLATES[INTENT_URGENT_HELP]
    )

    return f"""You are a senior Indian legal advisor — knowledgeable, clear, and human.
Respond in: {language}.
{risk_banner}
{enriched}
{entity_block}

CRITICAL LEGAL UPDATES (effective July 1, 2024):
- IPC → Bharatiya Nyaya Sanhita (BNS) 2023
- CrPC → Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023  
- Indian Evidence Act → Bharatiya Sakshya Adhiniyam (BSA) 2023
Always cite BNS/BNSS/BSA sections. When referencing old IPC/CrPC/IEA, note the new equivalent.

Ground rules:
- Follow the response format below exactly. Do not invent a different structure.
- Never open with "Based on the information provided" or similar filler phrases. Write directly.
- Never use internal labels like "Step 1 of Triage" or "Level 1". Just write.
- Statutory citation format: **Section X, Act Name** [source: filename.pdf]
- Case citation format: **Case Name (Year)** — [what it decided]
- Only cite laws and cases from the retrieved content below, or that you are completely certain about.
- If uncertain, say "verify this with a lawyer" — never guess a section number.
- End every response with an italicised disclaimer:
  *This is legal information for general guidance only, not legal advice or counsel.
   For your specific situation, consult a qualified advocate.*

Domain-specific guidance for this query:
{active_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT: {intent}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{format_template}
{secondary_block}"""


# ---------------------------------------------------------------------------
# MAIN RESPONSE GENERATION
# ---------------------------------------------------------------------------
def get_legal_response(
    context: str,
    sources: list,
    query: str,
    language: str,
    domains: list,
    history: list,
    enriched_situation: str = "",
    intent: str = INTENT_URGENT_HELP,
    risk_level: str = "LOW",
    entities: dict = None,
    secondary_intent: str = None,
    local_terms: list = None,
) -> str:

    system_prompt = build_system_prompt(
        domains, language, enriched_situation,
        intent, risk_level, entities, secondary_intent, local_terms,
    )

    history_str = ""
    if history:
        history_str = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history[-6:]
        )

    source_hint = ""
    if sources:
        source_hint = (
            "Source files retrieved (use these exact filenames in citations):\n"
            + "\n".join(f"  - {s}" for s in sources)
        )

    prompt = f"""---
Conversation so far:
{history_str if history_str else "This is the first message."}

---
User question: "{query}"
Intent: {intent} | Risk level: {risk_level}
Legal areas identified: {", ".join(domains)}

---
Retrieved legal content:
{context if context.strip() else "No relevant content found in the knowledge base for this question."}

{source_hint}
---

Citation reminder:
- Statutory: **Section X, Act Name** [source: filename.pdf]
- Case: **Case Name (Year)** — [what it decided]
- Only cite from retrieved content above or laws you are completely certain about.
- Prefer BNS/BNSS/BSA over IPC/CrPC/IEA. Note old section when mapping.

Write the response now, following the format specified in the system prompt."""

    return clean_response(
        llm_generate(
            prompt,
            system=system_prompt,
            model=PRIMARY_MODEL,
            max_tokens=5000,
            temperature=0.25,
        )
    )


# ---------------------------------------------------------------------------
# DOCUMENT ANALYSIS
# ---------------------------------------------------------------------------
def extract_text_from_file(file) -> str | None:
    filename = file.filename.lower()
    try:
        if filename.endswith(".pdf"):
            data = file.read()
            text = ""
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    pt = page.get_text("text").strip()
                    if pt:
                        text += pt + "\n"
                    else:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text += pytesseract.image_to_string(img) + "\n"
            return text
        elif filename.endswith(".txt"):
            return file.read().decode("utf-8")
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            return pytesseract.image_to_string(Image.open(file.stream))
        return None
    except Exception as e:
        logging.error(f"File extraction error: {e}")
        return None


def get_document_analysis_response(
    doc_text: str,
    query: str,
    language: str,
    context: str,
    sources: list,
    domains: list,
) -> str:
    # Determine DOC intent from query
    verify_keywords = ["valid", "legal", "compliant", "verify", "check", "is this",
                       "safe to sign", "can i sign", "is it correct"]
    doc_intent = INTENT_DOC_VERIFY if any(w in query.lower() for w in verify_keywords) else INTENT_DOC_ANALYSE

    system = build_system_prompt(
        domains, language, enriched_context="", intent=doc_intent,
    )

    source_hint = ""
    if sources:
        source_hint = "Retrieved source files:\n" + "\n".join(f"  - {s}" for s in sources)

    prompt = f"""Legal areas: {", ".join(domains)}.
User question: "{query}"
Intent: {doc_intent}

--- User's document (first 6000 characters) ---
{doc_text[:6000]}

--- Retrieved legal content ---
{context}

{source_hint}

Write the response following the {doc_intent} format from the system prompt."""

    return clean_response(
        llm_generate(
            prompt,
            system=system,
            model=PRIMARY_MODEL,
            max_tokens=5000,
            temperature=0.25,
        )
    )


# ---------------------------------------------------------------------------
# AGENTIC ROUTER — updated with preprocessing + risk scoring
# ---------------------------------------------------------------------------
def route_query(user_message: str, language: str = "en") -> str:
    global diagnostic_state
    detected_lang = detect_language(user_message, frontend_lang=language)
    lang_instr    = language_instruction(detected_lang)
    logging.info(f"Language — frontend: {language} | detected: {detected_lang} ({lang_instr})")

    # If a diagnostic session is active, use the language stored when it began
    if diagnostic_state["active"]:
        # Update language in case user switched mid-session (unlikely but safe)
        diagnostic_state["language"] = detected_lang
        return handle_diagnostic_answer(user_message, detected_lang)

    # Step 1: Preprocess (Hinglish normalization)
    normalized_query, local_terms = preprocess_query(user_message)

    # Step 2: Fast keyword-based risk scoring
    risk_level = score_risk(normalized_query)
    logging.info(f"Risk level: {risk_level}")

    # Step 3: Router intent (CHITCHAT / LEGAL_SEARCH / DOC_ANALYSIS)
    classify_prompt = f"""Read the user input and classify into exactly one category.
Reply with only the category word.

Categories:
- CHITCHAT: Greetings, casual conversation, thanks, hello
- LEGAL_SEARCH: Any legal question, rights, laws, procedures, advice, disputes
- DOC_ANALYSIS: User wants to analyse or review an uploaded document

Input: "{user_message}"
Category:"""

    try:
        raw_intent = llm_generate(
            classify_prompt, model=FAST_MODEL, max_tokens=10, temperature=0.05
        )
        router_intent = raw_intent.strip().upper() if raw_intent and not raw_intent.startswith("Service") else "LEGAL_SEARCH"
    except Exception as e:
        logging.error(f"Router error: {e}")
        router_intent = "LEGAL_SEARCH"

    logging.info(f"Router intent: {router_intent}")

    if "CHITCHAT" in router_intent:
        return llm_generate(
            f"Reply warmly and briefly in {lang_instr} as an Indian legal assistant. "
            f"Let the user know you can help with Indian legal questions. "
            f"The user said: {user_message}",
            model=FAST_MODEL,
            max_tokens=150,
            temperature=0.7,
        )

    elif "DOC" in router_intent:
        # Generate the doc-upload prompt in the detected language
        doc_prompt = (
            f"You are an Indian legal assistant. "
            f"Tell the user in {lang_instr} that they should upload their document using the "
            f"file upload button, and ask whether they want you to explain what it says "
            f"or verify whether it is legally valid."
        )
        return llm_generate(doc_prompt, model=FAST_MODEL, max_tokens=120, temperature=0.3)

    else:
        # HIGH risk: skip vagueness check, go straight to answer
        if risk_level == "HIGH":
            domains = detect_domains(normalized_query)
            return _generate_full_answer(
                normalized_query, detected_lang, domains,
                risk_level=risk_level, local_terms=local_terms,
            )

        domains = detect_domains(normalized_query)

        if is_query_vague(normalized_query, risk_level):
            diagnostic_state.update({
                "active":         True,
                "rounds_done":    0,
                "original_query": user_message,
                "collected_info": [],
                "domains":        domains,
                "language":       detected_lang,   # store for MCQ continuity
            })
            mcq = generate_mcq(user_message, domains, [], detected_lang, round_num=1)
            diagnostic_state["rounds_done"] = 1
            return mcq

        return _generate_full_answer(
            normalized_query, detected_lang, domains,
            risk_level=risk_level, local_terms=local_terms,
        )


def handle_diagnostic_answer(user_answer: str, language: str) -> str:
    global diagnostic_state

    fact = classify_mcq_answer(
        user_answer,
        diagnostic_state["original_query"],
        diagnostic_state["domains"],
    )
    diagnostic_state["collected_info"].append(fact)
    logging.info(f"Diagnostic fact collected: {fact}")

    rounds_done = diagnostic_state["rounds_done"]

    if rounds_done < MCQ_ROUNDS_REQUIRED:
        mcq = generate_mcq(
            diagnostic_state["original_query"],
            diagnostic_state["domains"],
            diagnostic_state["collected_info"],
            language,
            round_num=rounds_done + 1,
        )
        diagnostic_state["rounds_done"] += 1
        return mcq

    enriched = (
        f"Original question: {diagnostic_state['original_query']}\n"
        + "\n".join(diagnostic_state["collected_info"])
    )
    full_query = (
        f"{diagnostic_state['original_query']}. "
        f"Additional context: {'. '.join(diagnostic_state['collected_info'])}"
    )
    domains = diagnostic_state["domains"]

    diagnostic_state.update({
        "active": False, "rounds_done": 0,
        "original_query": "", "collected_info": [], "domains": [],
    })

    return _generate_full_answer(
        full_query, language, domains, enriched_situation=enriched,
    )


def _generate_full_answer(
    query: str,
    language: str,
    domains: list,
    enriched_situation: str = "",
    risk_level: str = "LOW",
    local_terms: list = None,
) -> str:
    # Detect full intent context (skipped for HIGH risk — already set)
    intent_ctx = detect_intent_and_risk(query, risk_level)

    context, sources, detected_domains = retrieve_context(query, language)
    all_domains = list(set(domains + detected_domains))

    return get_legal_response(
        context, sources, query, language, all_domains, chat_history,
        enriched_situation=enriched_situation,
        intent=intent_ctx["primary_intent"],
        risk_level=risk_level,
        entities=intent_ctx["entities"],
        secondary_intent=intent_ctx["secondary_intent"],
        local_terms=local_terms or [],
    )


# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

def _cors_preflight():
    r = make_response()
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    return r

@app.route("/")
def home():
    return send_from_directory(".", "minimal_chat.html")

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return _cors_preflight()
    global chat_history
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "No message provided."}), 400
        initialize_ai()
        message  = data["message"]
        language = data.get("language", "en")
        answer   = route_query(message, language)

        chat_history.append(HumanMessage(content=message))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({"response": answer})
    except Exception as e:
        logging.error(f"/chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-doc", methods=["POST", "OPTIONS"])
def analyze_doc():
    if request.method == "OPTIONS":
        return _cors_preflight()
    global chat_history
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        initialize_ai()
        file     = request.files["file"]
        message  = request.form.get("message", "Please analyse this document.")
        language = request.form.get("language", "en")

        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        doc_text = extract_text_from_file(file)
        if doc_text is None:
            return jsonify({"error": "Unsupported file type. Use PDF, TXT, PNG, or JPG."}), 400
        if not doc_text.strip():
            return jsonify({"error": "The document appears to be empty or unreadable."}), 400

        context, sources, domains = retrieve_context(message, language)
        answer = get_document_analysis_response(
            doc_text, message, language, context, sources, domains
        )

        chat_history.append(HumanMessage(content=f"[Uploaded: {file.filename}] {message}"))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({"response": answer})
    except Exception as e:
        logging.error(f"/analyze-doc error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST", "OPTIONS"])
def clear_history():
    if request.method == "OPTIONS":
        return _cors_preflight()
    global chat_history, diagnostic_state
    chat_history = []
    diagnostic_state.update({
        "active": False, "rounds_done": 0,
        "original_query": "", "collected_info": [], "domains": [],
    })
    return jsonify({"status": "Session cleared."})

@app.route("/domains", methods=["GET"])
def list_domains():
    return jsonify({"domains": list(DOMAIN_MAP.keys())})

@app.route("/intents", methods=["GET"])
def list_intents():
    return jsonify({
        "intents": ALL_INTENTS,
        "categories": {
            "A_emergency_procedural": [INTENT_URGENT_HELP, INTENT_POLICE_INTERACTION, INTENT_LEGAL_PROCEDURE],
            "B_substantive_strategy": [INTENT_LEGAL_EXPLANATION, INTENT_LAW_COMPARISON, INTENT_PENALTY_INFO, INTENT_CASE_SUMMARY, INTENT_LEGAL_STRATEGY],
            "C_civil_rights_domain":  [INTENT_RIGHTS_CHECK, INTENT_EVIDENCE_GUIDANCE, INTENT_JURISDICTION_CHECK, INTENT_LIMITATION_CHECK],
            "D_document_operations":  [INTENT_DOC_DRAFTING, INTENT_DOC_ANALYSE, INTENT_DOC_VERIFY],
        }
    })

@app.route("/diagnostic-status", methods=["GET"])
def diagnostic_status():
    return jsonify({
        "diagnostic_active": diagnostic_state["active"],
        "rounds_done":       diagnostic_state["rounds_done"],
        "rounds_required":   MCQ_ROUNDS_REQUIRED,
        "original_query":    diagnostic_state["original_query"],
        "collected_facts":   diagnostic_state["collected_info"],
    })

# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)