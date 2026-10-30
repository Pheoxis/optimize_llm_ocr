"""Checksum helpers for Polish / international identifiers."""

from __future__ import annotations


def pesel_checksum(digits: str) -> bool:
    if len(digits) != 11 or not digits.isdigit():
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(int(digits[i]) * weights[i] for i in range(10))
    check = (10 - (total % 10)) % 10
    return check == int(digits[10])


def nip_checksum(digits: str) -> bool:
    """Polish NIP — 10 digits."""
    if len(digits) != 10 or not digits.isdigit():
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    check = total % 11
    if check == 10:
        return False
    return check == int(digits[9])


def regon9_checksum(digits: str) -> bool:
    if len(digits) != 9 or not digits.isdigit():
        return False
    weights = (8, 9, 2, 3, 4, 5, 6, 7)
    total = sum(int(digits[i]) * weights[i] for i in range(8))
    check = total % 11
    if check == 10:
        check = 0
    return check == int(digits[8])


def regon14_checksum(digits: str) -> bool:
    if len(digits) != 14 or not digits.isdigit():
        return False
    if not regon9_checksum(digits[:9]):
        return False
    weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
    total = sum(int(digits[i]) * weights[i] for i in range(13))
    check = total % 11
    if check == 10:
        check = 0
    return check == int(digits[13])


def iban_checksum(iban: str) -> bool:
    raw = iban.replace(" ", "").upper()
    if len(raw) < 15 or len(raw) > 34:
        return False
    if not raw[:2].isalpha() or not raw[2:4].isdigit():
        return False
    if not raw[4:].isalnum():
        return False
    rearranged = raw[4:] + raw[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif "A" <= ch <= "Z":
            numeric += str(ord(ch) - 55)
        else:
            return False
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


# ISO 3779 transliteration (letters I, O, Q are not used in VINs).
_VIN_TRANSLITERATION: dict[str, int] = {
    **{c: i for i, c in enumerate("0123456789")},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}


def vin_charset_ok(vin: str) -> bool:
    if len(vin) != 17:
        return False
    for ch in vin:
        if ch not in _VIN_TRANSLITERATION:
            return False
    return True


def vin_check_digit_ok(vin: str) -> bool:
    """ISO 3779 check digit at position 9 (index 8)."""
    if not vin_charset_ok(vin):
        return False
    weights = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    total = 0
    for i, ch in enumerate(vin):
        if i == 8:
            continue
        total += _VIN_TRANSLITERATION[ch] * weights[i]
    remainder = total % 11
    if remainder == 10:
        expected = "X"
    else:
        expected = str(remainder)
    return vin[8].upper() == expected
