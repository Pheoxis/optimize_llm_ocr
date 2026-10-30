"""
Lightweight PL / EN word sets for OCR noise classification.

Embedded frozensets — no external dictionary files, no network, O(1) lookup.
Covers common document vocabulary (invoices, forms, postal, insurance).
"""

from __future__ import annotations

# ── Polish ────────────────────────────────────────────────────────────────────

PL_WORDS: frozenset[str] = frozenset({
    # document types
    "faktura", "paragon", "rachunek", "potwierdzenie", "polisa", "umowa",
    "list", "pismo", "wniosek", "formularz", "dokument", "oryginał", "oryginal",
    "kopia", "duplikat", "wzór", "wzor",
    # parties
    "sprzedawca", "nabywca", "płatnik", "platnik", "odbiorca", "nadawca",
    "wystawca", "kontrahent", "klient", "ubezpieczony", "ubezpieczający",
    "ubezpieczajacy", "agent", "broker", "dyrektor", "zarząd", "zarzad",
    # address / contact
    "adres", "ulica", "ul", "miasto", "kod", "pocztowy", "poczta", "skrytka",
    "telefon", "tel", "fax", "email", "strona", "www", "kontakt",
    # amounts / tax
    "kwota", "suma", "razem", "netto", "brutto", "vat", "podatek", "opłata",
    "oplata", "pobranie", "wpłata", "wplata", "saldo", "należność", "naleznosc",
    "zapłacono", "zaplacono", "do", "zapłaty", "zaplaty", "rabat", "procent",
    "zł", "zl", "gr", "pln", "grosz", "grosze",
    # identifiers (labels, not values)
    "nip", "regon", "pesel", "krs", "iban", "swift", "nr", "numer", "data",
    "termin", "płatności", "platnosci", "waluta", "konto", "bank",
    # dates / time
    "styczeń", "styczen", "luty", "marzec", "kwiecień", "kwiecien", "maj",
    "czerwiec", "lipiec", "sierpień", "sierpien", "wrzesień", "wrzesien",
    "październik", "pazdziernik", "listopad", "grudzień", "grudzien",
    "dzień", "dzien", "miesiąc", "miesiac", "rok", "godzina",
    # postal / insurance
    "przesyłka", "przesylka", "list", "polecony", "priorytet", "nadanie",
    "doręczenie", "doreczenie", "ubezpieczenie", "składka", "skladka",
    "polisa", "szkoda", "odszkodowanie", "ryzyko", "pokrycie",
    # common verbs / adjectives
    "jest", "być", "byc", "ma", "mają", "maja", "proszę", "prosze",
    "szanowni", "państwo", "panstwo", "poważaniem", "powazaniem",
    "dotyczy", "informacja", "informujemy", "niniejszym", "potwierdzamy",
    "zawarte", "obowiązuje", "obowiazuje", "ważne", "wazne", "ważny", "wazny",
    # form labels
    "imię", "imie", "nazwisko", "pesel", "data", "urodzenia", "miejscowość",
    "miejscowosc", "województwo", "wojewodztwo", "kraj", "polska", "polski",
    "tak", "nie", "podpis", "pieczęć", "pieczec", "pieczątka", "pieczatka",
    # cities (common on docs)
    "warszawa", "kraków", "krakow", "wrocław", "wroclaw", "poznań", "poznan",
    "gdańsk", "gdansk", "łódź", "lodz", "szczecin", "lublin", "katowice",
    # misc document
    "towar", "usługa", "usluga", "pozycja", "lp", "ilość", "ilosc", "j.m",
    "cena", "wartość", "wartosc", "jednostkowa", "razem", "słownie", "slownie",
    "ksef", "wizualizacja", "qr", "kod", "kreskowy",
    # short allowed
    "i", "w", "z", "o", "u", "na", "do", "od", "po", "we", "za", "ze",
    "nr", "np", "ul", "pl", "up", "fv", "fp",
})

# ── English ───────────────────────────────────────────────────────────────────

EN_WORDS: frozenset[str] = frozenset({
    "invoice", "receipt", "bill", "statement", "document", "original", "copy",
    "seller", "buyer", "vendor", "customer", "client", "recipient", "sender",
    "address", "street", "city", "postal", "code", "zip", "phone", "email",
    "amount", "total", "subtotal", "net", "gross", "tax", "vat", "discount",
    "payment", "paid", "due", "balance", "currency", "usd", "eur", "gbp",
    "date", "time", "number", "no", "ref", "reference", "account", "bank",
    "iban", "swift", "routing", "sort",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "item", "quantity", "qty", "price", "unit", "description", "service",
    "product", "goods", "shipping", "delivery", "tracking", "order",
    "please", "dear", "sir", "madam", "regards", "sincerely", "thank",
    "company", "limited", "ltd", "inc", "corp", "llc", "gmbh",
    "name", "signature", "signed", "authorized", "certified",
    "insurance", "policy", "premium", "coverage", "claim", "risk",
    "yes", "no", "true", "false", "from", "to", "the", "and", "for",
    "vat", "tin", "id", "ssn", "ein",
})

# Combined lookup (casefolded keys stored lowercase)
ALL_KNOWN: frozenset[str] = PL_WORDS | EN_WORDS
