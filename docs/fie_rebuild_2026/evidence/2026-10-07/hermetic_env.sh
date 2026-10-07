# Source this to get a CI-equivalent environment that never uses real secrets.
# Every key that exists in the repo .env is pre-set (empty) so load_dotenv(),
# which does not override existing variables, cannot inject a real credential.
REPO="c:/Users/ASUS/Desktop/Failure_Intelligence_System"
while IFS= read -r line; do
  key="${line%%=*}"
  case "$key" in ''|\#*) continue;; esac
  export "$key="
done < "$REPO/.env"
# Values CI sets explicitly (.github/workflows/ci.yml)
export MONGODB_URI="mongodb://localhost:27017"
export GROQ_API_KEY="fake-key-for-ci"
export JWT_SECRET_KEY="ci-secret-key-minimum-32-chars-long"
export ADMIN_EMAIL="ci@example.com"
export JWT_ALGORITHM="HS256"
export JWT_EXPIRE_HOURS="24"
export MONGODB_DB_NAME="fie_database"
export GROQ_ENABLED="true"
export OLLAMA_ENABLED="false"
export SERPER_ENABLED="false"
unset OLLAMA_MODELS GROQ_MODELS GROQ_FAST_MODEL CORS_ALLOWED_ORIGINS
# Keep side effects out of the repo and the home directory
export FIE_NO_TELEMETRY=1
export FIE_FEEDBACK_PATH="$SP/flagged_events.jsonl"
export FIE_NO_AUTO_DOWNLOAD=1
export PYTHONDONTWRITEBYTECODE=1
export PY="/c/Users/ASUS/anaconda3/envs/failure-engine/python.exe"
