import os
import re
import json
import secrets
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_sock import Sock
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)

# Browser frontend + Flask backend
CORS(app, supports_credentials=True)
sock = Sock(app)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound")

TECHVANTA_TEAM_NAME = os.getenv("TECHVANTA_TEAM_NAME", "Techvanta")
TECHVANTA_TEAM_PASSWORD = os.getenv("TECHVANTA_TEAM_PASSWORD")

MAX_CHAT_LENGTH = 5000
CHAT_WINDOW_SECONDS = 24 * 60 * 60
CHAT_CLIENTS = {}  # team_id -> set of active WebSocket connections
CHAT_LOCK = __import__("threading").Lock()


if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing in .env")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is missing in .env")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

if not TECHVANTA_TEAM_PASSWORD:
    raise RuntimeError("TECHVANTA_TEAM_PASSWORD is missing in .env")


supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value, max_length=None):
    if value is None:
        return ""

    value = str(value).strip()

    if max_length:
        value = value[:max_length]

    return value


def normalize_team_name(value):
    return re.sub(r"\s+", " ", clean_text(value)).strip().lower()


def research_excerpt(text, max_chars=12000):
    """Keep AI comparison requests bounded without truncating stored research."""
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[...middle omitted for AI comparison...]\n\n"
        + text[-tail:]
    )


def valid_username(username):
    return bool(
        re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username)
    )


def response_data(response, default=None):
    """
    Safely extract Supabase response.data.

    This prevents:
        'NoneType' object has no attribute 'data'
    """
    if response is None:
        return default

    data = getattr(response, "data", None)

    if data is None:
        return default

    return data


def supabase_execute(operation, label="Supabase operation"):
    """
    Execute a Supabase operation safely and give a useful error.
    """
    try:
        response = operation.execute()

        if response is None:
            raise RuntimeError(
                f"{label} returned no response. "
                "Check Supabase URL, key, database tables and permissions."
            )

        return response

    except Exception as e:
        print(f"{label} ERROR:", repr(e))
        raise


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    try:
        response = supabase_execute(
            supabase
            .table("users")
            .select(
                "id,username,display_name,team_id,"
                "research_score,research_count,"
                "useful_research_count,upgraded_research_count,"
                "time_spent_seconds,role,last_active_at,created_at"
            )
            .eq("id", user_id)
            .maybe_single(),
            "Get current user"
        )

        user = response_data(response)

        if not user:
            print("Current user was not found:", user_id)
            session.clear()
            return None

        return user

    except Exception as e:
        print("get_current_user error:", repr(e))
        return None


def get_current_team_id():
    user = get_current_user()

    if not user:
        return None

    return user.get("team_id")


# ============================================================
# AUTH DECORATOR
# ============================================================

def login_required(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        user = get_current_user()

        if not user:
            return jsonify({
                "success": False,
                "error": "Authentication required"
            }), 401

        return func(*args, **kwargs)

    return wrapper


# ============================================================
# SCORE
# ============================================================

def calculate_research_score(
    decision,
    confidence=0,
    has_source=False,
    time_seconds=0
):
    decision = clean_text(decision).upper()

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0

    try:
        time_seconds = int(time_seconds)
    except Exception:
        time_seconds = 0

    score = 0

    if decision == "NEW":
        score += 20

    elif decision == "UPGRADE":
        score += 30

    elif decision == "USEFUL":
        score += 15

    confidence_bonus = round(
        max(0, min(confidence, 100)) * 0.15
    )

    score += confidence_bonus

    if has_source:
        score += 10

    # 1 point for every 10 minutes, maximum 10
    time_bonus = min(
        10,
        max(0, time_seconds // 600)
    )

    score += time_bonus

    return int(score)


# ============================================================
# GROQ
# ============================================================

def groq_chat(messages, temperature=0.2):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature
        )

        choices = getattr(response, "choices", None)

        if not choices:
            raise RuntimeError("Groq returned no choices.")

        message = getattr(choices[0], "message", None)

        if message is None:
            raise RuntimeError("Groq returned no message.")

        content = getattr(message, "content", None)

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()

    except Exception as e:
        print("Groq error:", repr(e))
        error_text = str(e)
        # Do not impose an application-level size restriction.
        # If Groq itself rejects a request, return Groq's actual error.
        raise RuntimeError(f"Groq request failed: {e}")


# ============================================================
# EXISTING RESEARCH
# ============================================================

def get_existing_research(team_id, search_text="", limit=15):

    try:
        response = supabase_execute(
            supabase
            .table("research")
            .select(
                "id,team_id,user_id,topic,subtopic,finding,source,"
                "source_date,version,status,confidence,"
                "research_score,supersedes_id,created_at"
            )
            .eq("team_id", team_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(limit),
            "Get research"
        )

        rows = response_data(response, []) or []

        if not search_text:
            return rows

        words = [
            w.lower()
            for w in re.findall(r"\w+", search_text)
            if len(w) > 2
        ]

        if not words:
            return rows

        scored = []

        for row in rows:
            combined = (
                f"{row.get('topic', '')} "
                f"{row.get('subtopic', '')} "
                f"{row.get('finding', '')}"
            ).lower()

            matches = sum(
                1 for word in words if word in combined
            )

            if matches:
                scored.append((matches, row))

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            row for _, row in scored[:limit]
        ]

    except Exception as e:
        print("get_existing_research error:", repr(e))
        return []


# ============================================================
# RESEARCH EVALUATION
# ============================================================

def evaluate_research(team_id, user_question, new_information):

    existing = get_existing_research(
        team_id,
        user_question,
        limit=10
    )

    if existing:
        parts = []

        for item in existing:
            finding_text = research_excerpt(item.get("finding"), 2500)
            source_text = research_excerpt(item.get("source"), 500)
            parts.append(
                f"ID: {item.get('id')}\n"
                f"Topic: {item.get('topic')}\n"
                f"Subtopic: {item.get('subtopic')}\n"
                f"Finding: {finding_text}\n"
                f"Source: {source_text}\n"
                f"Version: {item.get('version')}\n"
                f"Confidence: {item.get('confidence')}"
            )

        existing_text = "\n".join(parts)

    else:
        existing_text = "NO EXISTING RESEARCH FOUND."

    # The full research can still be stored in Supabase, but only the
    # evidence needed for comparison is sent to Groq.
    new_information = research_excerpt(new_information, 20000)
    user_question = research_excerpt(user_question, 4000)

    prompt = f"""
You are the research quality evaluator for Team Techvanta.

The target is SIH problem statement 090.

The database must NOT contain every AI response.
Only strong, useful and meaningful research should be stored.

USER QUESTION:
{user_question}

NEW INFORMATION:
{new_information}

CURRENT TEAM KNOWLEDGE:
{existing_text}

Compare the new information with the existing team knowledge.

Choose exactly one decision:

NEW
UPGRADE
DUPLICATE
REJECT

NEW:
Important information that is not already present.

UPGRADE:
Information that is better, newer, more precise, more complete,
or more useful than an existing finding.

DUPLICATE:
Essentially the same information already exists.

REJECT:
Low quality, unsupported, irrelevant or unreliable information.

Return ONLY valid JSON in this format:

{{
  "decision": "NEW",
  "reason": "short explanation",
  "topic": "topic",
  "subtopic": "subtopic",
  "improved_finding": "best concise research finding",
  "source": "source URL or source name",
  "confidence": 0,
  "source_quality": "LOW"
}}

Confidence must be 0-100.

Do not invent sources.
If no source is supplied, leave source empty.
"""

    return groq_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict research quality evaluator. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )


def parse_json_response(text):
    if not text:
        return None

    text = str(text).strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except Exception:
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    return None


# ============================================================
# UPDATE SCORE
# ============================================================

def update_user_activity(
    user_id,
    score=0,
    time_seconds=0,
    research_count=0,
    useful_count=0,
    upgraded_count=0
):
    """
    First tries the database RPC.

    If the RPC is unavailable, it falls back to directly
    updating the users table.
    """

    try:
        response = supabase_execute(
            supabase.rpc(
                "add_research_activity",
                {
                    "p_user_id": user_id,
                    "p_score": score,
                    "p_time_seconds": time_seconds,
                    "p_research_count": research_count,
                    "p_useful_count": useful_count,
                    "p_upgraded_count": upgraded_count
                }
            ),
            "Update research activity RPC"
        )

        return True

    except Exception as rpc_error:
        print(
            "RPC unavailable; using manual score update:",
            repr(rpc_error)
        )

    try:
        response = supabase_execute(
            supabase
            .table("users")
            .select(
                "research_score,research_count,"
                "useful_research_count,upgraded_research_count,"
                "time_spent_seconds"
            )
            .eq("id", user_id)
            .maybe_single(),
            "Get user score"
        )

        current = response_data(response)

        if not current:
            raise RuntimeError(
                "Could not find user while updating score."
            )

        update = {
            "research_score": float(
                current.get("research_score") or 0
            ) + float(score),

            "research_count": int(
                current.get("research_count") or 0
            ) + int(research_count),

            "useful_research_count": int(
                current.get("useful_research_count") or 0
            ) + int(useful_count),

            "upgraded_research_count": int(
                current.get("upgraded_research_count") or 0
            ) + int(upgraded_count),

            "time_spent_seconds": int(
                current.get("time_spent_seconds") or 0
            ) + int(time_seconds)
        }

        supabase_execute(
            supabase
            .table("users")
            .update(update)
            .eq("id", user_id),
            "Manual user score update"
        )

        # Activity history is optional. Do not break the main action
        # if this table/RPC is unavailable.
        try:
            supabase_execute(
                supabase.table("activity").insert({
                    "user_id": user_id,
                    "score": score,
                    "time_seconds": time_seconds,
                    "research_count": research_count,
                    "useful_count": useful_count,
                    "upgraded_count": upgraded_count,
                    "created_at": now_iso()
                }),
                "Save activity"
            )
        except Exception as activity_error:
            print(
                "Activity history warning:",
                repr(activity_error)
            )

        return True

    except Exception as e:
        print("update_user_activity error:", repr(e))
        return False


# ============================================================
# HOME / HEALTH
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    return app.send_static_file("sw.js")


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "status": "online",
        "service": "Techvanta Project Hub",
        "database": "Supabase",
        "ai": "Groq",
        "model": GROQ_MODEL,
        "time": now_iso()
    })


# ============================================================
# REGISTER
# ============================================================

@app.route("/api/register", methods=["POST"])
def register():

    data = request.get_json(silent=True) or {}

    username = clean_text(
        data.get("username"),
        30
    )

    password = clean_text(
        data.get("password"),
        200
    )

    team_name = clean_text(
        data.get("team_name"),
        100
    )

    team_password = clean_text(
        data.get("team_password"),
        200
    )

    display_name = clean_text(
        data.get("display_name"),
        100
    )

    if not username:
        return jsonify({
            "success": False,
            "error": "Username is required."
        }), 400

    if not valid_username(username):
        return jsonify({
            "success": False,
            "error": (
                "Username must be 3-30 characters and use "
                "only letters, numbers, underscore, dot or hyphen."
            )
        }), 400

    if len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must contain at least 6 characters."
        }), 400

    if not team_name:
        return jsonify({
            "success": False,
            "error": "Team name is required."
        }), 400

    if not team_password:
        return jsonify({
            "success": False,
            "error": "Team password is required."
        }), 400

    if normalize_team_name(team_name) != normalize_team_name(
        TECHVANTA_TEAM_NAME
    ):
        return jsonify({
            "success": False,
            "error": "Account creation is restricted to Team Techvanta."
        }), 403

    if not secrets.compare_digest(
        team_password,
        TECHVANTA_TEAM_PASSWORD
    ):
        return jsonify({
            "success": False,
            "error": "Invalid Techvanta team password."
        }), 403

    try:
        # Do NOT use maybe_single() here. A normal select returns [] when
        # the username does not exist, which is exactly what registration needs.
        existing_response = supabase_execute(
            supabase
            .table("users")
            .select("id,username")
            .eq("username", username),
            "Check username"
        )

        existing_users = response_data(existing_response, []) or []

        if existing_users:
            return jsonify({
                "success": False,
                "error": "Username already exists."
            }), 409

        team_response = supabase_execute(
            supabase
            .table("teams")
            .select("id,team_name")
            .eq("team_name", TECHVANTA_TEAM_NAME)
            .maybe_single(),
            "Get Techvanta team"
        )

        team = response_data(team_response)

        if not team:
            return jsonify({
                "success": False,
                "error": (
                    "Techvanta team is not configured in Supabase. "
                    "Run the database schema and create the team first."
                )
            }), 500

        if not display_name:
            display_name = username

        new_user = {
            "username": username,
            "display_name": display_name,
            "password_hash": generate_password_hash(password),
            "team_id": team["id"],
            "research_score": 0,
            "research_count": 0,
            "useful_research_count": 0,
            "upgraded_research_count": 0,
            "time_spent_seconds": 0,
            "role": clean_text(data.get("role"), 80) or "Team Member",
            "last_active_at": now_iso()
        }

        result = supabase_execute(
            supabase.table("users").insert(new_user),
            "Create user"
        )

        created_rows = response_data(result, []) or []

        if not created_rows:
            return jsonify({
                "success": False,
                "error": "Account was not created."
            }), 500

        created_user = created_rows[0]

        return jsonify({
            "success": True,
            "message": "Techvanta account created successfully.",
            "user": {
                "id": created_user.get("id"),
                "username": created_user.get("username"),
                "display_name": created_user.get("display_name")
            }
        }), 201

    except Exception as e:
        print("Register error:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# LOGIN
# ============================================================

@app.route("/api/login", methods=["POST"])
def login():

    data = request.get_json(silent=True) or {}

    username = clean_text(
        data.get("username"),
        30
    )

    password = clean_text(
        data.get("password"),
        200
    )

    if not username or not password:
        return jsonify({
            "success": False,
            "error": "Username and password are required."
        }), 400

    try:
        result = supabase_execute(
            supabase
            .table("users")
            .select(
                "id,username,display_name,password_hash,team_id,"
                "research_score,research_count,"
                "useful_research_count,upgraded_research_count,"
                "time_spent_seconds,role,last_active_at"
            )
            .eq("username", username)
            .maybe_single(),
            "Login user lookup"
        )

        user = response_data(result)

        if not user:
            return jsonify({
                "success": False,
                "error": "Invalid username or password."
            }), 401

        stored_hash = user.get("password_hash")

        if not stored_hash:
            return jsonify({
                "success": False,
                "error": "This account has no valid password hash."
            }), 500

        if not check_password_hash(stored_hash, password):
            return jsonify({
                "success": False,
                "error": "Invalid username or password."
            }), 401

        try:
            supabase_execute(
                supabase.table("users").update({"last_active_at": now_iso()}).eq("id", user["id"]),
                "Update login activity"
            )
        except Exception as e:
            print("Login activity warning:", repr(e))

        session.clear()
        session["user_id"] = user["id"]

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "display_name": user.get("display_name"),
                "research_score": user.get("research_score", 0),
                "research_count": user.get("research_count", 0),
                "useful_research_count": user.get(
                    "useful_research_count", 0
                ),
                "upgraded_research_count": user.get(
                    "upgraded_research_count", 0
                ),
                "time_spent_seconds": user.get(
                    "time_spent_seconds", 0
                ),
                "role": user.get("role", "Team Member")
            }
        })

    except Exception as e:
        print("Login error:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# LOGOUT
# ============================================================

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()

    return jsonify({
        "success": True,
        "message": "Logged out successfully."
    })


# ============================================================
# CURRENT USER
# ============================================================

@app.route("/api/me")
@login_required
def me():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User not found."
        }), 404

    return jsonify({
        "success": True,
        "user": user
    })


# ============================================================
# TEAM CHAT — 24 HOUR WEBSOCKET WINDOW
# ============================================================

def chat_cutoff():
    return datetime.now(timezone.utc)


def cleanup_old_team_chat(team_id):
    """Maintain one shared 24-hour chat window for the whole team."""
    team_key = str(team_id)
    try:
        state_result = supabase_execute(
            supabase.table("team_chat_windows").select("team_id,started_at").eq("team_id", team_id).maybe_single(),
            "Get team chat window"
        )
        state = response_data(state_result)
        now = datetime.now(timezone.utc)
        reset = False

        if not state:
            supabase_execute(
                supabase.table("team_chat_windows").insert({"team_id": team_id, "started_at": now.isoformat()}),
                "Create team chat window"
            )
            next_reset = now.timestamp() + CHAT_WINDOW_SECONDS
        else:
            started = datetime.fromisoformat(str(state["started_at"]).replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if (now - started).total_seconds() >= CHAT_WINDOW_SECONDS:
                supabase_execute(
                    supabase.table("team_chat_messages").delete().eq("team_id", team_id),
                    "Reset 24 hour team chat"
                )
                supabase_execute(
                    supabase.table("team_chat_windows").update({"started_at": now.isoformat()}).eq("team_id", team_id),
                    "Start new team chat window"
                )
                reset = True
                next_reset = now.timestamp() + CHAT_WINDOW_SECONDS
            else:
                next_reset = started.timestamp() + CHAT_WINDOW_SECONDS

        if reset:
            broadcast_team_chat(team_key, {"type": "reset", "reason": "The 24-hour team chat window has reset.", "next_reset_at": datetime.fromtimestamp(next_reset, timezone.utc).isoformat()})
        return datetime.fromtimestamp(next_reset, timezone.utc).isoformat()
    except Exception as e:
        print("Chat window warning:", repr(e))
        # Fallback cleanup still prevents indefinite growth if the window table is unavailable.
        try:
            supabase_execute(
                supabase.table("team_chat_messages").delete().eq("team_id", team_id).lt("created_at", datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - CHAT_WINDOW_SECONDS, timezone.utc).isoformat()),
                "Fallback chat cleanup"
            )
        except Exception:
            pass
        return None


def broadcast_team_chat(team_id, payload):
    dead = []
    with CHAT_LOCK:
        clients = list(CHAT_CLIENTS.get(str(team_id), set()))
    for ws in clients:
        try:
            ws.send(json.dumps(payload))
        except Exception:
            dead.append(ws)
    if dead:
        with CHAT_LOCK:
            CHAT_CLIENTS.setdefault(str(team_id), set()).difference_update(dead)


def make_chat_message(user, message):
    return {
        "team_id": user["team_id"],
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "message": message,
        "created_at": now_iso()
    }


@sock.route('/ws/chat')
def websocket_chat(ws):
    """Native WebSocket team chat. Session cookies authenticate the socket."""
    user = get_current_user()
    if not user:
        try:
            ws.send(json.dumps({"type": "error", "error": "Authentication required"}))
        finally:
            ws.close()
        return

    team_key = str(user["team_id"])
    cleanup_old_team_chat(user["team_id"])
    with CHAT_LOCK:
        CHAT_CLIENTS.setdefault(team_key, set()).add(ws)

    try:
        ws.send(json.dumps({"type": "connected", "server_time": now_iso()}))
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                data = json.loads(raw)
            except Exception:
                ws.send(json.dumps({"type": "error", "error": "Invalid JSON."}))
                continue

            message = clean_text(data.get("message"), MAX_CHAT_LENGTH)
            if not message:
                ws.send(json.dumps({"type": "error", "error": "Message cannot be empty."}))
                continue

            cleanup_old_team_chat(user["team_id"])
            row = make_chat_message(user, message)
            try:
                result = supabase_execute(
                    supabase.table("team_chat_messages").insert(row),
                    "Save WebSocket team chat"
                )
                saved = response_data(result, []) or []
                if saved:
                    row = saved[0]
            except Exception as e:
                ws.send(json.dumps({"type": "error", "error": str(e)}))
                continue

            broadcast_team_chat(team_key, {"type": "message", "message": row})
    finally:
        with CHAT_LOCK:
            CHAT_CLIENTS.setdefault(team_key, set()).discard(ws)


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """HTTP fallback for environments where WebSocket is unavailable."""
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message"), MAX_CHAT_LENGTH)
    if not message:
        return jsonify({"success": False, "error": "Message cannot be empty."}), 400
    cleanup_old_team_chat(user["team_id"])
    row = make_chat_message(user, message)
    try:
        result = supabase_execute(
            supabase.table("team_chat_messages").insert(row),
            "Save team chat fallback"
        )
        saved = response_data(result, []) or []
        if saved:
            row = saved[0]
        broadcast_team_chat(str(user["team_id"]), {"type": "message", "message": row})
        return jsonify({"success": True, "message": row})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/history")
@login_required
def chat_history():
    user = get_current_user()
    next_reset_at = cleanup_old_team_chat(user["team_id"])
    try:
        result = supabase_execute(
            supabase.table("team_chat_messages")
            .select("id,user_id,username,display_name,message,created_at")
            .eq("team_id", user["team_id"])
            .order("created_at", desc=False)
            .limit(500),
            "Get 24 hour team chat"
        )
        return jsonify({
            "success": True,
            "messages": response_data(result, []) or [],
            "window_hours": 24,
            "next_reset_at": next_reset_at
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# ADD RESEARCH
# ============================================================


@app.route("/api/research", methods=["POST"])
@login_required
def add_research():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    data = request.get_json(silent=True) or {}

    topic = clean_text(data.get("topic"), 300)
    subtopic = clean_text(data.get("subtopic"), 300)
    finding = clean_text(data.get("finding"))
    source = clean_text(data.get("source"), 1000)
    source_date = clean_text(data.get("source_date"), 100)

    try:
        time_spent_seconds = int(
            data.get("time_spent_seconds", 0)
        )
    except Exception:
        time_spent_seconds = 0

    time_spent_seconds = max(
        0,
        min(time_spent_seconds, 86400)
    )

    if not topic:
        return jsonify({
            "success": False,
            "error": "Topic is required."
        }), 400

    if not finding:
        return jsonify({
            "success": False,
            "error": "Finding is required."
        }), 400

    try:
        evaluation_raw = evaluate_research(
            user["team_id"],
            topic,
            finding
        )

        evaluation = parse_json_response(
            evaluation_raw
        )

        if not evaluation:
            return jsonify({
                "success": False,
                "error": (
                    "AI research evaluation could not be parsed."
                ),
                "raw_evaluation": evaluation_raw
            }), 500

        decision = clean_text(
            evaluation.get("decision"),
            30
        ).upper()

        if decision not in {
            "NEW",
            "UPGRADE",
            "DUPLICATE",
            "REJECT"
        }:
            decision = "REJECT"

        try:
            confidence = float(
                evaluation.get("confidence", 0)
            )
        except Exception:
            confidence = 0

        confidence = max(
            0,
            min(100, confidence)
        )

        improved_finding = clean_text(
            evaluation.get("improved_finding")
        ) or finding

        evaluated_topic = clean_text(
            evaluation.get("topic"),
            300
        ) or topic

        evaluated_subtopic = clean_text(
            evaluation.get("subtopic"),
            300
        ) or subtopic

        evaluated_source = clean_text(
            evaluation.get("source"),
            1000
        ) or source

        # -----------------------------------------------
        # DUPLICATE / REJECT
        # -----------------------------------------------

        if decision in {"DUPLICATE", "REJECT"}:

            update_user_activity(
                user["id"],
                score=0,
                time_seconds=time_spent_seconds,
                research_count=1,
                useful_count=0,
                upgraded_count=0
            )

            return jsonify({
                "success": True,
                "saved": False,
                "decision": decision,
                "evaluation": evaluation,
                "message": (
                    "This research was not added to the "
                    "shared knowledge base."
                )
            })

        # -----------------------------------------------
        # FIND EXISTING ACTIVE RESEARCH
        # -----------------------------------------------

        existing = get_existing_research(
            user["team_id"],
            topic,
            limit=20
        )

        supersedes_id = None
        next_version = 1

        if existing:
            matching = existing[0]

            if decision == "UPGRADE":
                supersedes_id = matching.get("id")

                try:
                    next_version = int(
                        matching.get("version") or 1
                    ) + 1
                except Exception:
                    next_version = 2

        # -----------------------------------------------
        # SCORE
        # -----------------------------------------------

        score = calculate_research_score(
            decision=decision,
            confidence=confidence,
            has_source=bool(evaluated_source),
            time_seconds=time_spent_seconds
        )

        # -----------------------------------------------
        # SAVE
        # -----------------------------------------------

        research_row = {
            "team_id": user["team_id"],
            "user_id": user["id"],
            "topic": evaluated_topic,
            "subtopic": evaluated_subtopic,
            "finding": improved_finding,
            "source": evaluated_source,
            "source_date": source_date,
            "version": next_version,
            "status": "active",
            "confidence": confidence,
            "research_score": score,
            "supersedes_id": supersedes_id,
            "created_at": now_iso()
        }

        inserted = supabase_execute(
            supabase.table("research").insert(research_row),
            "Save research"
        )

        inserted_rows = response_data(inserted, []) or []

        if not inserted_rows:
            raise RuntimeError(
                "Research insert returned no data."
            )

        saved_research = inserted_rows[0]

        # Automatically add a discovered source to the shared Resources
        # section. This keeps Research and Resources synchronized.
        if evaluated_source:
            source_url = evaluated_source.strip()
            if source_url.startswith(("http://", "https://")):
                try:
                    parsed = urlparse(source_url)
                    source_title = evaluated_topic or parsed.netloc
                    existing_source = supabase_execute(
                        supabase
                        .table("sources")
                        .select("id")
                        .eq("team_id", user["team_id"])
                        .eq("url", source_url),
                        "Check research source"
                    )
                    if not (response_data(existing_source, []) or []):
                        supabase_execute(
                            supabase.table("sources").insert({
                                "team_id": user["team_id"],
                                "user_id": user["id"],
                                "title": source_title,
                                "url": source_url,
                                "topic": evaluated_topic,
                                "source_type": "AI Research",
                                "added_by": user["username"],
                                "created_at": now_iso()
                            }),
                            "Save research source"
                        )
                except Exception as source_error:
                    print("Research source warning:", repr(source_error))

        # -----------------------------------------------
        # SUPERSEDE OLD VERSION
        # -----------------------------------------------

        if decision == "UPGRADE" and supersedes_id:

            try:
                supabase_execute(
                    supabase
                    .table("research")
                    .update({
                        "status": "superseded"
                    })
                    .eq("id", supersedes_id),
                    "Supersede old research"
                )
            except Exception as e:
                print(
                    "Supersede warning:",
                    repr(e)
                )

        upgraded_count = 1 if decision == "UPGRADE" else 0

        update_user_activity(
            user["id"],
            score=score,
            time_seconds=time_spent_seconds,
            research_count=1,
            useful_count=0,
            upgraded_count=upgraded_count
        )

        return jsonify({
            "success": True,
            "saved": True,
            "decision": decision,
            "score": score,
            "evaluation": evaluation,
            "research": saved_research,
            "message": (
                "Research added to the shared knowledge base."
            )
        })

    except Exception as e:
        print("Add research error:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# GET RESEARCH
# ============================================================

@app.route("/api/research", methods=["GET"])
@login_required
def get_research():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    query = clean_text(
        request.args.get("q"),
        300
    )

    try:
        limit = int(
            request.args.get("limit", 100)
        )
    except Exception:
        limit = 100

    limit = max(
        1,
        min(limit, 200)
    )

    try:
        result = supabase_execute(
            supabase
            .table("research")
            .select(
                "id,topic,subtopic,finding,source,"
                "source_date,version,status,confidence,"
                "research_score,supersedes_id,created_at,user_id,useful_count"
            )
            .eq("team_id", user["team_id"])
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(limit),
            "Get research list"
        )

        rows = response_data(result, []) or []

        if query:
            words = [
                w.lower()
                for w in re.findall(r"\w+", query)
                if len(w) > 2
            ]

            if words:
                filtered = []

                for row in rows:
                    text = (
                        f"{row.get('topic', '')} "
                        f"{row.get('subtopic', '')} "
                        f"{row.get('finding', '')}"
                    ).lower()

                    if all(
                        word in text
                        for word in words
                    ):
                        filtered.append(row)

                rows = filtered

        return jsonify({
            "success": True,
            "research": rows
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MARK RESEARCH USEFUL
# ============================================================

@app.route("/api/research/<research_id>/useful", methods=["POST"])
@login_required
def mark_research_useful(research_id):
    """Mark a research finding useful once per teammate.

    The useful score belongs to the researcher who created the finding,
    not to the person who clicked the button.
    """
    voter = get_current_user()

    if not voter:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    try:
        research_response = supabase_execute(
            supabase
            .table("research")
            .select("id,team_id,user_id,useful_count")
            .eq("id", research_id)
            .maybe_single(),
            "Get research for useful vote"
        )
        research = response_data(research_response)

        if not research:
            return jsonify({
                "success": False,
                "error": "Research not found."
            }), 404

        if research.get("team_id") != voter.get("team_id"):
            return jsonify({
                "success": False,
                "error": "Research does not belong to your team."
            }), 403

        # The unique(research_id,user_id) database constraint prevents
        # the same teammate from voting twice.
        existing_vote = supabase_execute(
            supabase
            .table("research_votes")
            .select("id")
            .eq("research_id", research_id)
            .eq("user_id", voter["id"]),
            "Check useful vote"
        )
        if response_data(existing_vote, []) or []:
            return jsonify({
                "success": True,
                "already_voted": True,
                "score_added": 0,
                "message": "You already marked this research useful."
            })

        supabase_execute(
            supabase.table("research_votes").insert({
                "research_id": research_id,
                "user_id": voter["id"],
                "vote_type": "useful",
                "created_at": now_iso()
            }),
            "Create useful vote"
        )

        author_id = research.get("user_id")
        new_useful_count = int(research.get("useful_count") or 0) + 1

        supabase_execute(
            supabase
            .table("research")
            .update({"useful_count": new_useful_count})
            .eq("id", research_id),
            "Update research useful count"
        )

        # +15 belongs to the author of the useful finding.
        if author_id:
            update_user_activity(
                author_id,
                score=15,
                time_seconds=0,
                research_count=0,
                useful_count=1,
                upgraded_count=0
            )

        return jsonify({
            "success": True,
            "already_voted": False,
            "score_added": 15,
            "useful_count": new_useful_count,
            "message": "Research marked useful. The researcher received +15."
        })

    except Exception as e:
        print("Useful research error:", repr(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# SOURCES
# ============================================================

@app.route("/api/sources", methods=["GET"])
@login_required
def get_sources():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    try:
        result = supabase_execute(
            supabase
            .table("sources")
            .select(
                "id,title,url,topic,source_type,added_by,created_at"
            )
            .eq("team_id", user["team_id"])
            .order("created_at", desc=True)
            .limit(200),
            "Get sources"
        )

        return jsonify({
            "success": True,
            "sources": response_data(result, []) or []
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/sources", methods=["POST"])
@login_required
def add_source():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    data = request.get_json(silent=True) or {}

    title = clean_text(data.get("title"), 300)
    url = clean_text(data.get("url"), 1000)
    topic = clean_text(data.get("topic"), 300)
    source_type = clean_text(
        data.get("source_type"),
        100
    )

    if not title or not url:
        return jsonify({
            "success": False,
            "error": "Title and URL are required."
        }), 400

    row = {
        "team_id": user["team_id"],
        "user_id": user["id"],
        "title": title,
        "url": url,
        "topic": topic,
        "source_type": source_type,
        "added_by": user["username"],
        "created_at": now_iso()
    }

    try:
        result = supabase_execute(
            supabase.table("sources").insert(row),
            "Add source"
        )

        rows = response_data(result, []) or []

        return jsonify({
            "success": True,
            "source": rows[0] if rows else None
        }), 201

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# LEADERBOARD
# ============================================================

def _recent_iso(value, hours=24):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() <= hours * 3600
    except Exception:
        return False


@app.route("/api/leaderboard")
@login_required
def leaderboard():
    """AI-ranked team leaderboard based on real project-work evidence."""
    user = get_current_user()
    team_id = user["team_id"]
    try:
        members = response_data(supabase_execute(
            supabase.table("users").select(
                "id,username,display_name,role,research_score,research_count,"
                "useful_research_count,upgraded_research_count,time_spent_seconds,last_active_at"
            ).eq("team_id", team_id), "Get leaderboard members"), []) or []

        tasks = response_data(supabase_execute(
            supabase.table("tasks").select(
                "id,created_by,assigned_to,title,status,priority,due_date,created_at,completed_at"
            ).eq("team_id", team_id).limit(500), "Get leaderboard tasks"), []) or []

        research = response_data(supabase_execute(
            supabase.table("research").select(
                "id,user_id,status,confidence,research_score,useful_count,version,created_at"
            ).eq("team_id", team_id).limit(500), "Get leaderboard research"), []) or []

        milestones = response_data(supabase_execute(
            supabase.table("milestones").select("id,created_by,status,progress,due_date,created_at").eq("team_id", team_id).limit(300), "Get leaderboard milestones"), []) or []
        resources = response_data(supabase_execute(
            supabase.table("sources").select("id,user_id,created_at").eq("team_id", team_id).limit(500), "Get leaderboard resources"), []) or []

        activity = []
        for m in members:
            rows = response_data(supabase_execute(
                supabase.table("activity").select(
                    "user_id,score,time_seconds,research_count,useful_count,upgraded_count,details,created_at"
                ).eq("user_id", m["id"]).order("created_at", desc=True).limit(200),
                f"Get activity for {m.get('username')}"), []) or []
            activity.extend(rows)

        now = datetime.now(timezone.utc)
        metrics = []
        for m in members:
            uid = str(m["id"])
            uname = m.get("username") or ""
            assigned = [t for t in tasks if str(t.get("assigned_to") or "") == uname]
            created = [t for t in tasks if str(t.get("created_by") or "") == uid]
            completed = [t for t in assigned if str(t.get("status") or "").upper() == "COMPLETED"]
            overdue = [t for t in assigned if str(t.get("due_date") or "")[:10] and str(t.get("due_date"))[:10] < now.date().isoformat() and str(t.get("status") or "").upper() != "COMPLETED"]
            on_time = 0
            completion_minutes = []
            for t in completed:
                if t.get("completed_at") and t.get("due_date"):
                    if str(t["completed_at"])[:10] <= str(t["due_date"])[:10]:
                        on_time += 1
                if t.get("created_at") and t.get("completed_at"):
                    try:
                        a = datetime.fromisoformat(str(t["created_at"]).replace("Z", "+00:00"))
                        b = datetime.fromisoformat(str(t["completed_at"]).replace("Z", "+00:00"))
                        completion_minutes.append(max(0, int((b-a).total_seconds()/60)))
                    except Exception:
                        pass
            rws = [r for r in research if str(r.get("user_id")) == uid]
            mws = [x for x in milestones if str(x.get("created_by")) == uid]
            sws = [x for x in resources if str(x.get("user_id")) == uid]
            aws = [a for a in activity if str(a.get("user_id")) == uid]
            last_24 = [a for a in aws if _recent_iso(a.get("created_at"), 24)]
            active_minutes = sum(int(a.get("time_seconds") or 0) for a in last_24) / 60
            research_useful = sum(int(r.get("useful_count") or 0) for r in rws)
            high_done = sum(1 for t in completed if str(t.get("priority") or "").upper() in {"HIGH", "URGENT"})
            avg_speed = round(sum(completion_minutes)/len(completion_minutes), 1) if completion_minutes else None
            metrics.append({
                "id": uid, "username": uname, "display_name": m.get("display_name") or uname,
                "role": m.get("role") or "Team Member",
                "assigned_tasks": len(assigned), "completed_tasks": len(completed),
                "completion_rate": round(len(completed)/len(assigned)*100, 1) if assigned else 0,
                "overdue_tasks": len(overdue), "on_time_completed": on_time,
                "high_priority_completed": high_done, "created_tasks": len(created),
                "research_added": len(rws), "useful_research": research_useful,
                "research_upgrades": sum(1 for r in rws if int(r.get("version") or 1) > 1),
                "research_quality_points": round(sum(float(r.get("research_score") or 0) for r in rws), 1),
                "milestones_created": len(mws),
                "milestone_progress_average": round(sum(int(x.get("progress") or 0) for x in mws)/len(mws), 1) if mws else 0,
                "resources_added": len(sws),
                "workspace_minutes_last_24h": round(active_minutes, 1),
                "workspace_minutes_total": round(int(m.get("time_spent_seconds") or 0)/60, 1),
                "avg_task_completion_minutes": avg_speed,
                "active_last_24h": bool(last_24),
                "last_active_at": m.get("last_active_at")
            })

        prompt = """You are the impartial performance analyst for Team Techvanta's SIH26090 artisan/handmade-products project.
Rank team members ONLY from the supplied measured project-work facts. Do not reward time alone. Evaluate dedication, responsibility, role ownership, task assignment and completion rate, quick working speed, on-time delivery, high-priority work, research quantity and quality, useful research, upgrades, resources, milestone contribution, recent active participation, and responsibility coverage. Avoid double-counting the same evidence.

Return ONLY JSON:
{"rankings":[{"username":"...","score":0,"rank_reason":"...","strengths":["..."],"improvement":"..."}]}
Score is 0-100. Higher means stronger overall project contribution. Use all members exactly once. Never invent facts."""
        ai_payload = json.dumps(metrics, ensure_ascii=False)
        try:
            raw = groq_chat([
                {"role":"system","content":prompt},
                {"role":"user","content":"Measured team facts:\n" + research_excerpt(ai_payload, 18000)}
            ], temperature=0.0)
            ranked = parse_json_response(raw)
            ranking_rows = ranked.get("rankings", []) if isinstance(ranked, dict) else []
        except Exception as e:
            print("Leaderboard Groq warning:", repr(e))
            ranking_rows = []

        by_username = {str(x.get("username")): x for x in metrics}
        final = []
        used = set()
        for item in ranking_rows:
            uname = str(item.get("username") or "")
            if uname in by_username and uname not in used:
                used.add(uname)
                base = by_username[uname].copy()
                base.update({"score": max(0, min(100, int(float(item.get("score", 0) or 0)))), "rank_reason": clean_text(item.get("rank_reason"), 500), "strengths": item.get("strengths") if isinstance(item.get("strengths"), list) else [], "improvement": clean_text(item.get("improvement"), 300)})
                final.append(base)
        # Safe deterministic fallback only if Groq is unavailable/invalid.
        if len(final) != len(metrics):
            remaining = [m for m in metrics if m["username"] not in used]
            def fallback_score(m):
                return min(100, round(
                    m["completion_rate"] * .28 +
                    min(m["on_time_completed"] / max(1, m["completed_tasks"]), 1) * 15 +
                    min(m["high_priority_completed"] * 4, 12) +
                    min(m["useful_research"] * 4, 12) +
                    min(m["research_added"] * 2, 8) +
                    min(m["workspace_minutes_last_24h"] / 30, 10) +
                    (5 if m["active_last_24h"] else 0)
                ))
            for m in sorted(remaining, key=fallback_score, reverse=True):
                final.append({**m, "score": fallback_score(m), "rank_reason": "Automatic evidence-based fallback because the AI ranking response was unavailable.", "strengths": [], "improvement": "Keep improving task ownership and on-time delivery."})
        final.sort(key=lambda x: x["score"], reverse=True)
        for i, row in enumerate(final, 1): row["rank"] = i
        return jsonify({"success": True, "leaderboard": final, "ranking_engine": "Groq + measured project evidence"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# TEAM
# ============================================================

@app.route("/api/team")
@login_required
def team():

    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    try:
        team_result = supabase_execute(
            supabase
            .table("teams")
            .select("id,team_name")
            .eq("id", user["team_id"])
            .maybe_single(),
            "Get team"
        )

        team_data = response_data(team_result)

        members_result = supabase_execute(
            supabase
            .table("users")
            .select(
                "id,username,display_name,role,last_active_at,research_score,"
                "research_count,useful_research_count,"
                "upgraded_research_count,time_spent_seconds"
            )
            .eq("team_id", user["team_id"])
            .order("research_score", desc=True),
            "Get team members"
        )

        return jsonify({
            "success": True,
            "team": team_data,
            "members": response_data(
                members_result,
                []
            ) or []
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# TIME TRACKING
# ============================================================

@app.route("/api/activity/time", methods=["POST"])
@login_required
def save_time():
    """Add genuine research-session time to the current member."""
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    data = request.get_json(silent=True) or {}

    try:
        seconds = int(data.get("seconds", 0))
    except Exception:
        seconds = 0

    # A heartbeat may be at most one hour per request, but there is no
    # artificial daily limit. The frontend should send small heartbeats.
    seconds = max(0, min(seconds, 3600))

    if seconds == 0:
        return jsonify({
            "success": True,
            "added_seconds": 0
        })

    try:
        current = get_current_user()
        if not current:
            return jsonify({
                "success": False,
                "error": "User not found."
            }), 401

        old_total = int(current.get("time_spent_seconds") or 0)
        new_total = old_total + seconds

        # Time is a small leaderboard contribution.  Award only the
        # newly crossed 10-minute buckets, so repeated heartbeats cannot
        # inflate the score. The overall time bonus is capped at +10.
        old_time_bonus = min(10, max(0, old_total // 600))
        new_time_bonus = min(10, max(0, new_total // 600))
        time_score_delta = max(0, new_time_bonus - old_time_bonus)

        # Use the same activity updater as research submissions so the
        # score, time total and activity history stay synchronized.
        try:
            supabase_execute(
                supabase.table("users").update({"last_active_at": now_iso()}).eq("id", user["id"]),
                "Update last active"
            )
        except Exception as e:
            print("last_active_at warning", repr(e))

        update_user_activity(
            user["id"],
            score=time_score_delta,
            time_seconds=seconds,
            research_count=0,
            useful_count=0,
            upgraded_count=0
        )

        # Keep an auditable history when the activity table exists.
        try:
            supabase_execute(
                supabase.table("activity").insert({
                    "user_id": user["id"],
                    "score": 0,
                    "time_seconds": seconds,
                    "research_count": 0,
                    "useful_count": 0,
                    "upgraded_count": 0,
                    "details": "Research session heartbeat",
                    "created_at": now_iso()
                }),
                "Save time activity"
            )
        except Exception as activity_error:
            print("Time activity warning:", repr(activity_error))

        return jsonify({
            "success": True,
            "added_seconds": seconds,
            "total_seconds": new_total,
            "time_score_added": time_score_delta,
            "time_spent_text": format_duration(new_total)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# ACTIVITY SUMMARY
# ============================================================

@app.route("/api/activity/me", methods=["GET"])
@login_required
def activity_me():
    """Return the current member's live contribution counters."""
    user = get_current_user()
    if not user:
        return jsonify({
            "success": False,
            "error": "User session is invalid."
        }), 401

    seconds = int(user.get("time_spent_seconds") or 0)
    return jsonify({
        "success": True,
        "activity": {
            "research_score": float(user.get("research_score") or 0),
            "research_count": int(user.get("research_count") or 0),
            "useful_research_count": int(
                user.get("useful_research_count") or 0
            ),
            "upgraded_research_count": int(
                user.get("upgraded_research_count") or 0
            ),
            "time_spent_seconds": seconds,
            "time_spent_text": format_duration(seconds)
        }
    })


def format_duration(seconds):
    try:
        seconds = max(0, int(seconds))
    except Exception:
        seconds = 0

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"



# ============================================================
# PROJECT MANAGEMENT
# ============================================================

@app.route("/api/project/summary")
@login_required
def project_summary():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "User session is invalid."}), 401
    team_id = user["team_id"]
    try:
        tasks_r = supabase_execute(supabase.table("tasks").select("id,status,priority,due_date,assigned_to,title,created_at,completed_at").eq("team_id", team_id).order("created_at", desc=True).limit(300), "Get project tasks")
        tasks = response_data(tasks_r, []) or []
        milestones_r = supabase_execute(supabase.table("milestones").select("id,title,description,due_date,status,progress,created_at").eq("team_id", team_id).order("due_date", desc=False).limit(100), "Get milestones")
        milestones = response_data(milestones_r, []) or []
        members_r = supabase_execute(supabase.table("users").select("id,username,display_name,research_score,research_count,useful_research_count,upgraded_research_count,time_spent_seconds").eq("team_id", team_id).order("research_score", desc=True), "Get project members")
        members = response_data(members_r, []) or []
        total = len(tasks)
        completed = sum(1 for t in tasks if str(t.get("status", "")).upper() == "COMPLETED")
        in_progress = sum(1 for t in tasks if str(t.get("status", "")).upper() == "IN PROGRESS")
        today = datetime.now(timezone.utc).date().isoformat()
        overdue = sum(1 for t in tasks if str(t.get("due_date") or "")[:10] and str(t.get("due_date") or "")[:10] < today and str(t.get("status", "")).upper() != "COMPLETED")
        progress = round((completed / total) * 100) if total else 0
        return jsonify({"success": True, "summary": {"total_tasks": total, "completed_tasks": completed, "in_progress_tasks": in_progress, "overdue_tasks": overdue, "progress": progress, "members": len(members)}, "tasks": tasks[:20], "milestones": milestones[:10], "members": members})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/tasks", methods=["GET", "POST"])
@login_required
def tasks_api():
    user = get_current_user()
    if not user: return jsonify({"success": False, "error": "Authentication required"}), 401
    team_id = user["team_id"]
    if request.method == "GET":
        try:
            status = clean_text(request.args.get("status"), 30)
            query = supabase.table("tasks").select("*").eq("team_id", team_id).order("created_at", desc=True).limit(300)
            if status: query = query.eq("status", status)
            rows = response_data(supabase_execute(query, "Get tasks"), []) or []
            return jsonify({"success": True, "tasks": rows})
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    data = request.get_json(silent=True) or {}
    title = clean_text(data.get("title"), 200)
    description = clean_text(data.get("description"), 5000)
    assigned_to = clean_text(data.get("assigned_to"), 100)
    priority = clean_text(data.get("priority"), 20).upper() or "MEDIUM"
    status = clean_text(data.get("status"), 30).upper() or "TODO"
    due_date = clean_text(data.get("due_date"), 40) or None
    if not title: return jsonify({"success": False, "error": "Task title is required."}), 400
    if priority not in {"LOW", "MEDIUM", "HIGH", "URGENT"}: priority = "MEDIUM"
    if status not in {"TODO", "IN PROGRESS", "REVIEW", "COMPLETED", "BLOCKED"}: status = "TODO"
    try:
        row = {"team_id": team_id, "created_by": user["id"], "title": title, "description": description, "assigned_to": assigned_to or None, "priority": priority, "status": status, "due_date": due_date, "created_at": now_iso()}
        if status == "COMPLETED": row["completed_at"] = now_iso()
        result = supabase_execute(supabase.table("tasks").insert(row), "Create task")
        rows = response_data(result, []) or []
        return jsonify({"success": True, "task": rows[0] if rows else row}), 201
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/tasks/<task_id>", methods=["PUT", "DELETE"])
@login_required
def task_detail(task_id):
    user = get_current_user()
    if not user: return jsonify({"success": False, "error": "Authentication required"}), 401
    try:
        check = response_data(supabase_execute(supabase.table("tasks").select("id,team_id,status").eq("id", task_id).maybe_single(), "Get task"))
        if not check: return jsonify({"success": False, "error": "Task not found."}), 404
        if check.get("team_id") != user["team_id"]: return jsonify({"success": False, "error": "Task does not belong to your team."}), 403
        if request.method == "DELETE":
            supabase_execute(supabase.table("tasks").delete().eq("id", task_id), "Delete task")
            return jsonify({"success": True})
        data = request.get_json(silent=True) or {}
        updates = {}
        for key, max_len in [("title",200),("description",5000),("assigned_to",100),("priority",20),("status",30),("due_date",40)]:
            if key in data: updates[key] = clean_text(data.get(key), max_len) or None
        if "priority" in updates and updates["priority"] not in {"LOW","MEDIUM","HIGH","URGENT"}: updates["priority"] = "MEDIUM"
        if "status" in updates:
            updates["status"] = updates["status"].upper()
            if updates["status"] not in {"TODO","IN PROGRESS","REVIEW","COMPLETED","BLOCKED"}: updates["status"] = "TODO"
            updates["completed_at"] = now_iso() if updates["status"] == "COMPLETED" else None
        if not updates: return jsonify({"success": False, "error": "Nothing to update."}), 400
        result = supabase_execute(supabase.table("tasks").update(updates).eq("id", task_id), "Update task")
        rows = response_data(result, []) or []
        return jsonify({"success": True, "task": rows[0] if rows else updates})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/milestones", methods=["GET", "POST"])
@login_required
def milestones_api():
    user = get_current_user()
    if not user: return jsonify({"success": False, "error": "Authentication required"}), 401
    if request.method == "GET":
        try:
            rows = response_data(supabase_execute(supabase.table("milestones").select("*").eq("team_id", user["team_id"]).order("due_date", desc=False), "Get milestones"), []) or []
            return jsonify({"success": True, "milestones": rows})
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    data = request.get_json(silent=True) or {}
    title = clean_text(data.get("title"), 200)
    if not title: return jsonify({"success": False, "error": "Milestone title is required."}), 400
    try: progress = max(0, min(100, int(data.get("progress", 0) or 0)))
    except Exception: progress = 0
    row = {"team_id": user["team_id"], "created_by": user["id"], "title": title, "description": clean_text(data.get("description"), 3000), "due_date": clean_text(data.get("due_date"), 40) or None, "status": clean_text(data.get("status"), 30).upper() or "PLANNED", "progress": progress, "created_at": now_iso()}
    try:
        result = supabase_execute(supabase.table("milestones").insert(row), "Create milestone")
        rows = response_data(result, []) or []
        return jsonify({"success": True, "milestone": rows[0] if rows else row}), 201
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/notifications")
@login_required
def notifications_api():
    user = get_current_user()
    if not user: return jsonify({"success": False, "error": "Authentication required"}), 401
    try:
        rows = response_data(supabase_execute(supabase.table("notifications").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(50), "Get notifications"), []) or []
        return jsonify({"success": True, "notifications": rows})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ai/history")
@login_required
def ai_history():
    user = get_current_user()
    try:
        result = supabase_execute(
            supabase.table("chat_messages").select("id,user_id,username,message,response,created_at")
            .eq("team_id", user["team_id"]).order("created_at", desc=False).limit(100),
            "Get AI history"
        )
        return jsonify({"success": True, "messages": response_data(result, []) or []})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/project/ai", methods=["POST"])
@login_required
def project_ai():
    user = get_current_user()
    if not user: return jsonify({"success": False, "error": "Authentication required"}), 401
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message"), MAX_CHAT_LENGTH)
    if not message: return jsonify({"success": False, "error": "Message cannot be empty."}), 400
    try:
        tasks = response_data(supabase_execute(supabase.table("tasks").select("title,status,priority,due_date,assigned_to").eq("team_id", user["team_id"]).order("created_at", desc=True).limit(30), "Get AI task context"), []) or []
        research = get_existing_research(user["team_id"], message, limit=8)
        context = "TASKS:\n" + "\n".join([f"{t.get('title')} | {t.get('status')} | {t.get('priority')} | due {t.get('due_date')} | {t.get('assigned_to')}" for t in tasks])
        context += "\n\nRELEVANT TEAM KNOWLEDGE:\n" + "\n".join([f"{r.get('topic')}: {research_excerpt(r.get('finding'), 1000)}" for r in research])
        prompt = """You are the private AI project assistant for Team Techvanta. STRICTLY support only SIH26090, the artisan/handmade-products problem, and this team's implementation. If the question is unrelated, refuse briefly. Do not behave as a general chatbot. Use the supplied project context. Never invent team facts.

""" + research_excerpt(context, 12000)
        answer = groq_chat([{"role":"system","content":prompt},{"role":"user","content":message}], temperature=0.2)
        try:
            supabase_execute(supabase.table("chat_messages").insert({"team_id":user["team_id"],"user_id":user["id"],"username":user["username"],"message":message,"response":answer,"created_at":now_iso()}), "Save project AI chat")
        except Exception as e: print("AI chat save warning", repr(e))
        return jsonify({"success": True, "response": answer})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("             SIH RESEARCH AI")
    print("=" * 60)
    print()
    print("Team:", TECHVANTA_TEAM_NAME)
    print("Database: Supabase")
    print("AI:", GROQ_MODEL)
    print()
    print("Server:")
    print("http://127.0.0.1:5000")
    print()
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True
    )
