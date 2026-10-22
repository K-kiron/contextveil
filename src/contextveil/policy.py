"""Policy for explicit structured fields and review requirements."""

import re
from dataclasses import dataclass, field

from .models import ContextVeilError

FIELD_TYPES = {
    "name": "PERSON", "full_name": "PERSON", "first_name": "PERSON", "last_name": "PERSON",
    "patient_name": "PERSON", "customer_name": "PERSON", "employee_name": "PERSON",
    "email": "EMAIL", "email_address": "EMAIL", "phone": "PHONE", "phone_number": "PHONE",
    "mobile": "PHONE", "ssn": "SSN", "social_security_number": "SSN",
    "address": "ADDRESS", "street_address": "ADDRESS", "date_of_birth": "DATE", "dob": "DATE",
    "passport": "ID", "passport_number": "ID", "account_number": "ID", "patient_id": "ID",
    "customer_id": "ID", "employee_id": "ID", "credit_card": "CREDIT_CARD", "card_number": "CREDIT_CARD",
    "password": "SECRET", "api_key": "SECRET", "secret": "SECRET", "access_token": "SECRET",
    "ip_address": "IP_ADDRESS", "url": "URL",
}
LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,31}\Z")


def field_key(value: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value.strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[\s-]+", "_", value).lower()


@dataclass
class Policy:
    field_types: dict[str, str] = field(default_factory=lambda: FIELD_TYPES.copy())
    require_contextual: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.require_contextual, bool) or not isinstance(self.field_types, dict):
            raise ContextVeilError("Invalid policy configuration.")
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not LABEL.fullmatch(v)
               for k, v in self.field_types.items()):
            raise ContextVeilError("Field policy requires nonempty names and uppercase entity labels.")
        self.field_types = {field_key(k): v for k, v in self.field_types.items()}

    @classmethod
    def from_dict(cls, value: dict) -> "Policy":
        if not isinstance(value, dict) or set(value) - {"field_types", "require_contextual"}:
            raise ContextVeilError("Unknown or invalid policy option.")
        fields = FIELD_TYPES.copy()
        overrides = value.get("field_types", {})
        if not isinstance(overrides, dict):
            raise ContextVeilError("Field policy must be a mapping.")
        fields.update(overrides)
        return cls(fields, value.get("require_contextual", True))
