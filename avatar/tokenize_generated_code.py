import io
import re
import tokenize


TOKEN_RE = re.compile(r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|0[xX][0-9A-Fa-f_]+|0[bB][01_]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[A-Za-z]*|[A-Za-z_$][A-Za-z0-9_$]*|>>>=|>>>|>>=|<<=|==|!=|<=|>=|\+\+|--|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|[{}()\[\];,.:?@~!+\-*/%<>=&|^]', re.S | re.M)
LITERAL_RE = re.compile(r"STRNEWLINE|TABSYMBOL|▁|\\.|0[xX][0-9A-Fa-f_]+|0[bB][01_]+|\.\d+[A-Za-z]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[A-Za-z]*|[A-Za-z]+|_+|>>>=|>>>|>>=|<<=|==|!=|<=|>=|\+\+|--|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|[{}()\[\];,.:?@~!+\-*/%<>=&|^]|[^\s]", re.S)


def clean_code(text):
    text = text.strip()
    if "```" in text:
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    return text.strip()


def literal_body_tokens(body):
    if body == "":
        return []
    if "▁" in body or "STRNEWLINE" in body or "TABSYMBOL" in body:
        body = re.sub(r"\s+", " ", body.strip())
        return [tok for tok in LITERAL_RE.findall(body) if tok.strip()]

    out = []
    pieces = re.split(r"(\s+)", body)
    for piece in pieces:
        if not piece:
            continue
        if piece.isspace():
            out.append("▁")
            continue
        for tok in LITERAL_RE.findall(piece):
            if tok.startswith("_") and len(tok) > 1:
                out.extend("_" for _ in tok)
            else:
                out.append(tok)
    return out


def literal_tokens(text):
    match = re.fullmatch(r'([rRuUbBfF]*)(["\'])(.*)\2', text, re.S)
    if not match:
        return [text]
    prefix = match.group(1)
    quote = match.group(2)
    body = match.group(3)
    if not prefix and body.isdigit():
        return [text]
    if not prefix and quote == "'" and len(body) == 1 and (body.isdigit() or body.isspace()):
        return [text]
    parts = list(prefix) if prefix else []
    return parts + [quote] + literal_body_tokens(body) + [quote]


def generic_tokens(code, keep_newlines=True):
    tokens = []
    for match in TOKEN_RE.finditer(code):
        text = match.group(0)
        if text.startswith("//") or text.startswith("/*"):
            continue
        tokens.extend(literal_tokens(text) if text[:1] in {'"', "'"} else [text])
    if not keep_newlines:
        return tokens

    lines = []
    for line in code.splitlines():
        line_tokens = generic_tokens(line, keep_newlines=False)
        if line_tokens:
            lines.extend(line_tokens + ["NEW_LINE"])
    return lines[:-1] if lines else tokens


def tokenize_python_code(code):
    out = []
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT):
                continue
            if tok.type == fstring_start:
                prefix, quote = tok.string[:-1], tok.string[-1]
                if prefix:
                    out.append(prefix)
                out.append(quote)
                continue
            elif tok.type == fstring_middle:
                out.extend(literal_body_tokens(tok.string))
                continue
            elif tok.type == fstring_end:
                out.append(tok.string)
                continue
            if tok.type == tokenize.NEWLINE:
                out.append("NEW_LINE")
            elif tok.type == tokenize.INDENT:
                out.append("INDENT")
            elif tok.type == tokenize.DEDENT:
                out.append("DEDENT")
            elif tok.type == tokenize.NL:
                continue
            elif tok.type == tokenize.STRING:
                out.extend(literal_tokens(tok.string))
            elif tok.string.strip():
                out.append(tok.string)
    except Exception:
        out = generic_tokens(code)
    return " ".join(out).strip()


def tokenize_java_code(code):
    return " ".join(generic_tokens(code, keep_newlines=False)).strip()


def postprocess(text, target_language):
    code = clean_code(text)
    if target_language == "python" and " NEW_LINE " in f" {code} ":
        return re.sub(r"\s+", " ", code.replace("\n", " ")).strip()
    return tokenize_python_code(code) if target_language == "python" else tokenize_java_code(code)
