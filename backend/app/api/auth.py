"""Auth endpoints: register, login, logout."""
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.db.database import SessionLocal
from app.db.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


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
        # Verify even when the user is missing would be ideal to avoid timing leaks; for
        # this scope a constant message is enough to avoid user enumeration.
        if user is None or not verify_password(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        return TokenResponse(access_token=create_access_token(str(user.id)))
    finally:
        db.close()


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    # Stateless JWT: the client discards the token. A server-side blocklist is the
    # documented optional extension; not implemented for this scope.
    return {"detail": "Logged out. Discard the token client-side."}
