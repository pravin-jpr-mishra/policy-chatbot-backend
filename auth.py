import os
import msal
import base64
import json
from dotenv import load_dotenv

load_dotenv()


CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # "development" or "production"

# Production URLs - set these in your environment variables
PRODUCTION_FRONTEND_URL = os.getenv("PRODUCTION_FRONTEND_URL", "https://your-app.vercel.app")

# Supported redirect URIs for different environments
REDIRECT_URIS = {
    # Local development
    "local_react": "http://localhost:3000",
    "local_streamlit": "http://localhost:8501/",
    # Production (Vercel)
    "production": PRODUCTION_FRONTEND_URL,
}

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["User.Read"]


def get_redirect_uri(origin: str = None) -> str:
    """Automatically determine redirect URI based on request origin or environment"""
    if origin:
        # Check for local development
        if ":3000" in origin:
            return REDIRECT_URIS["local_react"]
        elif ":8501" in origin:
            return REDIRECT_URIS["local_streamlit"]
        # Check for production (Vercel domain)
        elif "vercel.app" in origin or (PRODUCTION_FRONTEND_URL and PRODUCTION_FRONTEND_URL.replace("https://", "").replace("http://", "") in origin):
            return REDIRECT_URIS["production"]
    
    # If in production environment, default to production URL
    if ENVIRONMENT == "production":
        return REDIRECT_URIS["production"]
    
    # Default to local React for development
    return REDIRECT_URIS["local_react"]


def _build_msal_app():
    return msal.ConfidentialClientApplication(
        client_id=CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )


def get_login_url(origin: str = None):
    """Get login URL with automatic redirect URI detection"""
    redirect_uri = get_redirect_uri(origin)
    app = _build_msal_app()
    
    # Encode redirect URI in state so we know which one to use in callback
    state_data = {"redirect_uri": redirect_uri}
    state = base64.urlsafe_b64encode(json.dumps(state_data).encode()).decode()
    
    return app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        state=state,
        prompt="select_account",
    )


def process_login_response(query_params: dict, state: str = None):
    code = query_params.get("code") if hasattr(query_params, 'get') else query_params._params.get("code")
    if not code:
        print("No code in query params")
        return None

    # Extract redirect URI from state
    redirect_uri = REDIRECT_URIS["local_react"]  # default
    if state:
        try:
            state_data = json.loads(base64.urlsafe_b64decode(state).decode())
            redirect_uri = state_data.get("redirect_uri", redirect_uri)
            print(f"Using redirect_uri from state: {redirect_uri}")
        except Exception as e:
            print(f"Failed to decode state: {e}")
    
    print(f"Processing login with redirect_uri: {redirect_uri}")
    
    app = _build_msal_app()
    result = app.acquire_token_by_authorization_code(
        code=code,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )

    if "id_token_claims" not in result:
        print(f"Token acquisition failed: {result.get('error_description', result)}")
        return None

    claims = result["id_token_claims"]
    print(f"Got claims for user: {claims.get('preferred_username', 'unknown')}")

    # STRICT tenant validation (org-only access)
    if claims.get("tid") != TENANT_ID:
        print(f"Tenant mismatch: {claims.get('tid')} != {TENANT_ID}")
        return None

    return claims
