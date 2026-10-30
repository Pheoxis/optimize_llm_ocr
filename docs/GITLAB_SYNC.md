# Weryfikacja synchronizacji GitLab

Ten plik potwierdza, że gałąź `main` na GitLabie zawiera aktualną dokumentację.

| Pole | Wartość |
|------|---------|
| Ostatni commit dokumentacji | `6c98b9f` |
| Poprzedni duży merge kodu | `e3958c5` |
| Zaktualizowano | 2026-05-29 |

## Co powinno być w repozytorium

- Katalog `docs/` z plikami: `README.md`, `PIPELINE.md`, `ARCHITECTURE.md`, `API.md`, `SCHEMA.md`, `ENV_REFERENCE.md`, `TROUBLESHOOTING.md`, `SAMPLES.md`, `STAN_PROJEKTU.md`
- W `README.md` (root) — baner WIP i link do `docs/README.md`

## Sprawdzenie w GitLab (UI)

1. Otwórz: https://100.127.177.52/Kamil/optimize_llm_ocr/-/commits/main  
2. Najnowszy commit powinien zawierać w tytule: **Expand project documentation** lub **docs: mark GitLab sync**
3. Plik: https://100.127.177.52/Kamil/optimize_llm_ocr/-/blob/main/docs/GITLAB_SYNC.md

## Sprawdzenie lokalnie

```bash
git fetch origin
git log origin/main -1 --oneline
# oczekiwane: 6c98b9f lub nowszy z "GitLab sync"
```
