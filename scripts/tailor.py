import argparse
import json
import os
import re
from typing import List, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


def escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def extract_block(content: str, start_pattern: str, end_pattern: str) -> Tuple[int, int, str]:
    start_match = re.search(start_pattern, content, flags=re.S)
    if not start_match:
        return -1, -1, ""
    start_index = start_match.end()
    end_match = re.search(end_pattern, content[start_index:], flags=re.S)
    if not end_match:
        return -1, -1, ""
    end_index = start_index + end_match.start()
    return start_index, end_index, content[start_index:end_index]


class TailorResponse(BaseModel):
    summary: str = Field(..., description="1-3 sentence summary <= 350 chars")
    skills: List[str] = Field(..., description="6-12 skills")
    top_role_bullets: List[str] = Field(..., description="4-6 bullets for latest role")


def replace_section(original: str, start_pattern: str, end_pattern: str, replacement: str) -> str:
    start_idx, end_idx, _ = extract_block(original, start_pattern, end_pattern)
    if start_idx == -1:
        return original
    return original[:start_idx] + replacement + original[end_idx:]


def replace_summary(tex: str, summary: str) -> str:
    replacement = escape_tex(summary.strip()) + "\n\n"
    return replace_section(tex, r"\\section\{Summary\}\s*", r"\\section\{", replacement)


def replace_first_itemize(tex: str, section_pattern: str, items: List[str]) -> str:
    section_match = re.search(section_pattern, tex)
    if not section_match:
        return tex
    after_section = tex[section_match.end():]
    begin_match = re.search(r"\\begin\{itemize\}", after_section)
    if not begin_match:
        return tex
    begin_index = section_match.end() + begin_match.start()
    end_match = re.search(r"\\end\{itemize\}", tex[begin_index:])
    if not end_match:
        return tex
    end_index = begin_index + end_match.end()
    header = tex[:begin_index]
    footer = tex[end_index:]
    body = "\n".join(
        f"\\item {escape_tex(item.strip())}"
        for item in items
        if item.strip()
    )
    replacement = (
        "\\begin{itemize}[nosep,after=\\strut, leftmargin=1em, itemsep=3pt,label=$\\bullet$]\n"
        f"{body}\n"
        "\\end{itemize}"
    )
    return header + replacement + footer


def extract_resume_components(tex: str):
    _, _, summary_block = extract_block(tex, r"\\section\{Summary\}\s*", r"\\section\{")
    summary_text = summary_block.strip()

    work_section_match = re.search(r"\\section\{Work Experience\}", tex)
    latest_bullets = []
    if work_section_match:
        after_section = tex[work_section_match.end():]
        begin_match = re.search(r"\\begin\{itemize\}", after_section)
        if begin_match:
            begin_index = work_section_match.end() + begin_match.end()
            end_match = re.search(r"\\end\{itemize\}", tex[begin_index:])
            if end_match:
                list_block = tex[begin_index: begin_index + end_match.start()]
                latest_bullets = [
                    re.sub(r"^\\item\s*", "", line).strip()
                    for line in list_block.splitlines()
                    if line.strip().startswith("\\item")
                ]

    skills_match = re.search(r"\\section\{Core Technical Skills\}", tex)
    skills_block = ""
    if skills_match:
        after_section = tex[skills_match.end():]
        begin_match = re.search(r"\\begin\{itemize\}", after_section)
        if begin_match:
            begin_index = skills_match.end() + begin_match.end()
            end_match = re.search(r"\\end\{itemize\}", tex[begin_index:])
            if end_match:
                skills_block = tex[begin_index: begin_index + end_match.start()]

    return summary_text, latest_bullets, skills_block


def build_prompt(jd: str, summary: str, bullets: List[str], skills_block: str, extra: str) -> List[dict]:
    system = (
        "You are CareerForgeAI, tailoring only the resume summary, latest role bullets, "
        "and skills. Keep outputs truthful, action-oriented, and aligned with the job description. "
        "Return strict JSON with keys: summary, skills, top_role_bullets. "
        "Plain text only, no LaTeX."
    )

    bullets_text = "\n- ".join([b.strip() for b in bullets]) if bullets else ""

    user = f"""Job Description:
```
{jd.strip()}
```

Current Summary:
```
{summary.strip()}
```

Latest Role Bullets:
```
- {bullets_text}
```

Current Skills block:
```
{skills_block.strip()}
```

Constraints:
- Summary: 1-3 sentences, <= 350 chars, truthful.
- Skills: 6-12 concise items.
- Latest role bullets: 4-6 bullets, one line each, action + measurable impact where supported.
- Use JD keywords only when they reflect real experience.
- Do not fabricate new tools or responsibilities.
"""
    if extra:
        user += f"\nAdditional Instructions:\n{extra.strip()}\n"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_model(messages: List[dict]) -> TailorResponse:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return TailorResponse(**data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True)
    parser.add_argument("--instructions", default="")
    parser.add_argument("--input-tex", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    with open(args.input_tex, "r", encoding="utf-8") as fh:
        tex = fh.read()

    summary, latest_bullets, skills_block = extract_resume_components(tex)

    messages = build_prompt(
        jd=args.jd,
        summary=summary,
        bullets=latest_bullets,
        skills_block=skills_block,
        extra=args.instructions,
    )

    try:
        tailored = call_model(messages)
    except (json.JSONDecodeError, ValidationError) as error:
        raise RuntimeError(f"Failed to parse model response: {error}") from error

    updated = replace_summary(tex, tailored.summary)
    updated = replace_first_itemize(updated, r"\\section\{Work Experience\}", tailored.top_role_bullets)
    updated = replace_first_itemize(updated, r"\\section\{Core Technical Skills\}", tailored.skills)

    with open(args.output_tex, "w", encoding="utf-8") as fh:
        fh.write(updated)


if __name__ == "__main__":
    main()

