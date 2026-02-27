import os
import subprocess
import json
import time
import re
from dotenv import load_dotenv
import pyperclip
from google import genai
from google.genai import types

# ==========================
# Setup
# ==========================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: Please set your GEMINI_API_KEY in .env file")
    exit(1)

client = genai.Client(api_key=api_key)

MODEL_LITE = "gemini-2.5-flash-lite"


def safe_json_loads(raw_text, context_label=""):
    """
    Try to parse JSON; if that fails, try to extract the first {...} block.
    Return None on failure.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    print(f"[WARN] Failed to parse JSON for {context_label}")
    return None


def read_resume(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Make sure it's in the same folder :( ")
        exit(1)


# ==========================================
# 1. SIGNAL EXTRACTION (GENERALIZED)
# ==========================================

def extract_signal_weights(jd):
    print("\n[1/4] Extracting and weighting JD signals...")

    clusters = [
        "backend",
        "frontend",
        "fullstack",
        "data_ml",
        "cloud_devops",
        "testing_quality",
        "performance_scalability",
        "security_reliability",
        "product_user_focus",
        "teamwork_communication",
        "learning_growth",
        "domain_industry",
    ]

    cluster_list_str = ", ".join(clusters)

    prompt = f"""
You are an expert job description analyzer.

Analyze the following Job Description and identify how strongly it emphasizes each of these signal clusters:
[{cluster_list_str}]

Definitions:
- backend: server-side logic, APIs, databases, services
- frontend: UI, UX, web or mobile interfaces
- fullstack: end-to-end ownership across frontend and backend
- data_ml: data pipelines, analytics, ML, statistics, modeling
- cloud_devops: cloud platforms, CI/CD, containers, deployment
- testing_quality: testing, QA, reliability, validation
- performance_scalability: speed, latency, scaling, optimization
- security_reliability: security, robustness, fault tolerance
- product_user_focus: user needs, UX, customer impact, domain workflows
- teamwork_communication: collaboration, communication, culture, cross-functional work
- learning_growth: onboarding, mentorship, training, continuous improvement
- domain_industry: specific industry or domain (e.g., K-12, healthcare, finance, etc.)

Weighting Rules (1-10 scale):
- Required skills/core tasks: 8-10
- Day-to-day responsibilities: 5-7
- Preferred/Bonus skills: 3-4
- Not mentioned / irrelevant: 0-1

Return a STRICT JSON object with these keys:
- "signal_weights": object mapping cluster names to integer weights (0-10).
- "top_5_signals": list of the 5 highest-weighted clusters (strings).
- "top_keywords": list of the 10 most critical hard skills/technologies (strings).
- "domain_phrases": list of up to 5 short phrases describing industry, users, or mission
    (e.g., "K-12 schools", "students and teachers", "school administrators", "healthcare providers").

Job Description:
{jd}
"""

    response = client.models.generate_content(
        model=MODEL_LITE,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    data = safe_json_loads(response.text, context_label="JD signals")
    if not data:
        return None

    data.setdefault("signal_weights", {})
    data.setdefault("top_5_signals", [])
    data.setdefault("top_keywords", [])
    data.setdefault("domain_phrases", [])
    return data


# ==========================================
# 2. RESUME BULLET TAGGING
# ==========================================

def tag_resume_bullets(resume_content):
    print("[2/4] Semantically tagging resume bullets...")

    clusters = [
        "backend",
        "frontend",
        "fullstack",
        "data_ml",
        "cloud_devops",
        "testing_quality",
        "performance_scalability",
        "security_reliability",
        "product_user_focus",
        "teamwork_communication",
        "learning_growth",
        "domain_industry",
    ]
    cluster_list_str = ", ".join(clusters)

    prompt = f"""
You are analyzing a LaTeX resume that uses \\resumeItem{{...}} for bullet points.

Task:
1. Extract EVERY bullet point (the text inside \\resumeItem{{...}}).
2. For each bullet, assign 1 to 3 relevant signal clusters from this exact list:
   [{cluster_list_str}]

Return a STRICT JSON object with this shape:
{{
  "tagged_bullets": [
    {{
      "bullet_text": "The exact original text inside \\resumeItem{{...}}",
      "tags": ["cluster1", "cluster2"]
    }},
    ...
  ]
}}

LaTeX Resume:
{resume_content}
"""

    response = client.models.generate_content(
        model=MODEL_LITE,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    data = safe_json_loads(response.text, context_label="bullet tagging")
    if not data:
        return []

    tagged = data.get("tagged_bullets", [])
    if not isinstance(tagged, list):
        return []
    return tagged


# ==========================================
# 3. DYNAMIC SCORING & ADJUSTMENT
# ==========================================

def score_and_adjust_bullets(jd, jd_signals, tagged_bullets, resume_content):
    print("[3/4] Computing dynamic alignment scores and executing LaTeX adjustments...")

    scored_bullets = []
    weights = jd_signals.get("signal_weights", {}) or {}

    for bullet in tagged_bullets:
        tags = bullet.get("tags", []) or []
        score = sum(int(weights.get(tag, 0)) for tag in tags)
        scored_bullets.append(
            {
                "bullet_text": bullet.get("bullet_text", ""),
                "tags": tags,
                "alignment_score": score,
            }
        )

    scored_bullets.sort(key=lambda x: x["alignment_score"], reverse=True)
    bullets_context = json.dumps(scored_bullets, indent=2)

    top_signals = ", ".join(jd_signals.get("top_5_signals", []))
    top_keywords = ", ".join(jd_signals.get("top_keywords", []))
    domain_phrases = ", ".join(jd_signals.get("domain_phrases", []))

    prompt = f"""
You are an elite resume-tailoring engine.

I am providing:
- Scored resume bullets
- Job description signals and keywords
- Domain phrases (industry, users, mission)
- The raw LaTeX resume

Your task:
Rewrite the \\resumeItem{{...}} bullets in the LaTeX code to better align with the Job Description signals, while staying 100% truthful.

SCORING CONTEXT:
You will see a JSON list of bullets with:
- "bullet_text"
- "tags" (signal clusters)
- "alignment_score" (integer)

DYNAMIC SCORING DIRECTIVES:
- High-Scoring Bullets: Keep them, optionally move slightly higher within their section, and expand meaningful technical depth where useful.
- Low-Scoring Bullets: Compress aggressively (around 15–18 words), or reframe them to more strongly touch on a Top 5 Signal *only if authentic*.

KEYWORD & DOMAIN ALIGNMENT:
- Top 5 JD Signals: {top_signals}
- Top 10 Keywords: {top_keywords}
- Domain Phrases: {domain_phrases}

ENFORCEMENT:
- Ensure at least ONE strong bullet explicitly reflects EACH of the Top 5 Signals.
- When the Job Description clearly indicates a specific domain or user group
  (e.g., K-12 schools, students, teachers, administrators, healthcare providers),
  reflect that domain in 2–3 bullets where the connection is natural.
- If domain_phrases are provided, you may reuse 2–3 of them across Experience or Projects
  bullets **only when it is honest**. If the project was not actually in that domain, phrase it
  as “similar to how [domain users] do X” rather than claiming it is directly used by them.

USER & IMPACT FRAMING:
- Emphasize who benefits from the work (users, customers, classmates, instructors, stakeholders),
  especially if "product_user_focus" is a high-weight signal.
- When mentioning teamwork, communication, documentation, or learning, briefly indicate the context
  (e.g., with a research team, with store staff, for future developers).

AVOIDING GENERIC/BUZZWORDY PHRASES:
- Do NOT end bullets with abstract phrases like:
  - "supporting learning and growth"
  - "showcasing product user focus"
  - "demonstrating user focus"
  or similar vague statements.
- Instead, end bullets with a concrete, specific impact or use case, such as:
  - reduced errors or manual work
  - faster workflows or response times
  - clearer feedback or easier reviews for specific users.

NON-WORDY ENFORCEMENT:
- Target: 20–28 words per bullet. Hard max: 30 words.
- Syntax: Maximum 2 commas per bullet. Maximum 1 "and" per bullet.
- Tone: Remove filler ("leveraged", "utilized", "cutting-edge", "comprehensive"). Prefer concrete verbs.
- Formatting: Bold 2–3 highly relevant keywords in each bullet using \\textbf{{...}}.

LATEX SAFETY:
- Return ONLY raw LaTeX. No markdown, no explanations, no backticks.
- Preserve section order.
- Do NOT add or remove sections.
- Ensure "Technical Skills" and "Education" sections appear exactly once.
- Keep the same LaTeX structure (\\section, \\resumeSubheading, \\resumeItem, etc.).

Scored Bullets Context (JSON):
{bullets_context}

Current LaTeX Resume:
{resume_content}
"""

    response = client.models.generate_content(
        model=MODEL_LITE,
        contents=prompt,
    )

    clean_text = response.text or ""

    match = re.search(r"\\documentclass.*?\\end\{document\}", clean_text, re.DOTALL)
    if match:
        clean_text = match.group(0)

    clean_text = clean_text.replace("```latex", "").replace("```", "").strip()
    return clean_text


# ==========================================
# 4. LATEX COMPILATION
# ==========================================

def compile_latex(latex_content, company_name):
    tex_filename = "tailored_resume.tex"
    clean_cn = re.sub(r"[^A-Za-z0-9_-]+", "-", company_name.strip())
    jobname = f"resume_{clean_cn}"

    print(f"\n[4/4] Saving and Compiling LaTeX to {jobname}.pdf...")

    with open(tex_filename, "w", encoding="utf-8") as file:
        file.write(latex_content)

    try:
        process = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"-jobname={jobname}", tex_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if process.returncode == 0:
            print(f"Success! Your optimized resume is ready: {jobname}.pdf")
        else:
            print("Warning: pdflatex finished with potential formatting issues. Check the PDF.")

        for ext in [".aux", ".log", ".out"]:
            file_to_rm = f"{jobname}{ext}"
            if os.path.exists(file_to_rm):
                os.remove(file_to_rm)

    except FileNotFoundError:
        print("\nError: Could not find 'pdflatex' command. Is LaTeX installed and on PATH?")


# ==========================================
# 5. MAIN FLOW
# ==========================================

def main():
    print("Welcome to the Dynamic Signal Resume Optimization Engine")
    print("-" * 55)

    company_name = input("Enter company name (eg: Google): ").strip()
    print("-" * 55)

    print("Copy the Job Description to your Clipboard.")
    input("Press Enter here once copied...\n")

    jd = pyperclip.paste()
    if not jd.strip():
        print("Error: Clipboard is empty.")
        exit(1)

    print(f"Captured {len(jd.split())} words.\n")

    # Step 1: Extract Signals
    jd_signals = extract_signal_weights(jd)
    if not jd_signals:
        print("Error: Failed to extract JD signals.")
        exit(1)

    print("\n🎯 Dynamic JD Signals Extracted:")
    print(f" Top 5 Clusters: {', '.join(jd_signals.get('top_5_signals', []))}")
    print(f" Target Keywords: {', '.join(jd_signals.get('top_keywords', [])[:6])}...")
    if jd_signals.get("domain_phrases"):
        print(f" Domain Phrases: {', '.join(jd_signals.get('domain_phrases', []))}")

    print("\nPausing for 10 seconds for API Rate Limits...")
    time.sleep(10)

    # Step 2: Tag Bullets
    base_resume = read_resume("resume.tex")
    tagged_bullets = tag_resume_bullets(base_resume)

    if not tagged_bullets:
        print("[WARN] No tagged bullets returned. Tailoring will be more generic.")

    print("\nPausing for 10 seconds for API Rate Limits...")
    time.sleep(10)

    # Step 3: Score and Adjust
    tailored_resume = score_and_adjust_bullets(jd, jd_signals, tagged_bullets, base_resume)

    print("\n" + "=" * 20 + " PROPOSED CHANGES " + "=" * 20)
    print(tailored_resume)
    print("=" * 58 + "\n")

    while True:
        user_input = input("Compile PDF? (Y/N): ").strip().upper()
        if user_input == "Y":
            compile_latex(tailored_resume, company_name)
            break
        elif user_input == "N":
            print("Exiting without compilation.")
            break
        else:
            print("Type 'Y' or 'N'.")


if __name__ == "__main__":
    main()
