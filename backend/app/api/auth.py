"""Auth endpoints: register, login, logout."""
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.db.database import SessionLocal
from app.db.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

# Precomputed once at import: verified against on the missing-user branch so login takes
# roughly the same time whether or not the account exists (avoids timing-based user
# enumeration). (P1.8)
_DUMMY_HASH = hash_password("invalid-placeholder-password-never-matches")


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> UserOut:
    email = request.email.lower()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first() is not None:
            raise HTTPException(status_code=409, detail="Email already registered.")
        user = User(email=email, hashed_password=hash_password(request.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserOut.model_validate(user)
    finally:
        db.close()


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    email = request.email.lower()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        # Always run a hash verification (against a dummy hash when the user is missing) so
        # response time doesn't reveal whether the account exists.
        hashed = user.hashed_password if user is not None else _DUMMY_HASH
        password_ok = verify_password(request.password, hashed)
        if user is None or not password_ok:
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        return TokenResponse(access_token=create_access_token(str(user.id)))
    finally:
        db.close()


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    # Stateless JWT: the client discards the token. A server-side blocklist is the
    # documented optional extension; not implemented for this scope.
    return {"detail": "Logged out. Discard the token client-side."}
