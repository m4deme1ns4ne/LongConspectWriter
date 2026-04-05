def debug_bad_words(tokenizer) -> None:
    """Только для отладки. Вызывать вручную, не из рабочего кода."""
    for word in BAD_WORDS[:10]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        decoded = [tokenizer.decode([i]) for i in ids]
        print(f"{word!r:40} → ids={ids} → токены={decoded}")