from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import sys
import os

security_scheme = HTTPBearer()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecret")
JWT_ALGORITHM = "HS256"


def create_jwt_token(user_id: int, expires_delta: timedelta = timedelta(days=1)) -> str:
    """Create JWT token with string user_id"""
    expire = datetime.utcnow() + expires_delta
    # Ensure user_id is converted to string
    user_id_str = str(user_id)
    print(f"Creating token with user_id: {user_id_str}")

    to_encode = {
        "sub": user_id_str,  # Explicit string conversion
        "exp": expire,
        "type": "access"  # Add token type for extra validation
    }

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY,
                             algorithm=JWT_ALGORITHM)
    print(f"Token created successfully")
    return f"Bearer {encoded_jwt}"  # Add Bearer prefix here


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> int:
    """Verify JWT token and return user_id as integer"""
    print("\n=== Token Verification ===")
    token = credentials.credentials
    print(f"Received token: {token[:15]}...")
    sys.stdout.flush()

    try:
        # Remove Bearer prefix if present
        if token.startswith('Bearer '):
            token = token.replace('Bearer ', '')
            print("Removed 'Bearer ' prefix")

        # Decode and verify token
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_sub": True}  # Enforce subject verification
        )
        print("Decoded payload:", payload)
        sys.stdout.flush()

        # Validate payload structure
        if not isinstance(payload, dict):
            print("Error: Payload is not a dictionary")
            raise HTTPException(
                status_code=401,
                detail="Invalid token structure"
            )

        # Extract and validate subject claim
        user_id_str = payload.get("sub")
        if not user_id_str or not isinstance(user_id_str, str):
            print("Error: Invalid subject claim")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing user ID in token"
            )

        # Convert to integer
        try:
            user_id = int(user_id_str)
            print(f"Successfully verified token for user_id: {user_id}")
            return user_id
        except ValueError:
            print(
                f"Error: Could not convert user_id '{user_id_str}' to integer")
            raise HTTPException(
                status_code=401,
                detail="Invalid user ID format"
            )

    except JWTError as e:
        print(f"JWT Verification failed: {str(e)}")
        sys.stdout.flush()
        raise HTTPException(
            status_code=401,
            detail=f"Token verification failed: {str(e)}"
        )
