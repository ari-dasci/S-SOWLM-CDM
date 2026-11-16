"""The protocol tables: prompts, decoding parameters and model provenance.

    python -m utils.protocol london
    python -m utils.protocol madrid

Prompts are read from `config/experiments.yaml`, the file the runner reads.
The other two tables are rendered from snapshots of the serving runtime taken
when the data were collected and kept beside the archives (`raw/<corpus>/`),
because a model tag is mutable and cannot be queried again later:

    model_provenance.csv   per served model: tag, manifest digest,
                           architecture, parameter count, quantization and
                           context length (`ollama show`, `ollama list`), the
                           first and last run timestamp and the reasoning-trace
                           setting in its run records, the runtime version and
                           GPU driver, and the date of the snapshot.
    model_parameters.csv   London only: the Modelfile defaults
                           (`ollama show --parameters`) and the digest they
                           were read from. They are reported only for a system
                           whose collection digest is that one.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from .profiles import ROOT, load_profile

EXPERIMENTS = ROOT / "config" / "experiments.yaml"
SCHEMES = (
    ("SCORE", "single_score_v2_t1"),
    ("LABEL-7", "single_label_7c_t1"),
    ("LABEL-5", "single_label_5c_t1"),
    ("LABEL-5, $T = 0$", "single_label_5c_t0"),
)


def _tex(text: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        text = text.replace(a, b)
    return text


def prompts_to_latex(out_path: Path) -> None:
    config = yaml.safe_load(EXPERIMENTS.read_text())
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Prompts, verbatim from \texttt{config/experiments.yaml}. "
            r"The system prompt is empty for every scheme; \texttt{\{review\}} "
            r"is replaced by the review text. The $T = 0$ collection uses the "
            r"LABEL-5 prompt unchanged. Every call to a locally served model "
            r"sets the temperature shown, caps the reply at 2{,}048 tokens and "
            r"disables the reasoning trace (\texttt{think: false}), except where "
            r"Table~\ref{tab:decoding} records the setting as unrecorded.}"
        ),
        r"\label{tab:prompts}",
        r"\small",
        r"\begin{tabular}{lp{0.70\textwidth}r}",
        r"\toprule",
        r"\textbf{Scheme} & \textbf{User prompt} & \textbf{$T$} \\",
        r"\midrule",
    ]
    for label, key in SCHEMES:
        entry = config[key]
        prompt = entry["user_template"].replace("\n", " ").strip()
        lines.append(
            f"  {label} & \\texttt{{{_tex(prompt)}}} & {entry['temperature']:.1f}"
            r" \\"
        )
        assert entry["system_prompt"] == "", key
        assert entry["max_tokens"] == 2048 and entry["think"] is False, key
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def decoding_to_latex(
    params: pd.DataFrame, provenance: pd.DataFrame, out_path: Path
) -> None:
    """One row per served system: request-set and model-default parameters."""

    by_tag = params.set_index("tag")["parameters"].to_dict()
    captured_digest = params.set_index("tag")["digest"].to_dict()
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Decoding parameters. \emph{Request} is what the runner "
            r"set on every call; temperature, the 2{,}048-token cap and "
            r"\texttt{think} come from the scheme (Table~\ref{tab:prompts}). "
            r"\emph{Model defaults} are the Modelfile parameters the runtime "
            r"applied to everything else, as \texttt{ollama show --parameters} "
            r"reports them; they belong to the manifest, so they are given "
            r"only where the digest recorded at collection is still the one "
            r"installed, and are otherwise unrecorded. A request temperature "
            r"overrides the Modelfile's. The hosted reference used its "
            r"provider's defaults.}"
        ),
        r"\label{tab:decoding}",
        r"\small",
        r"\begin{tabular}{lllp{0.42\textwidth}}",
        r"\toprule",
        (
            r"\textbf{System} & \textbf{Digest} & \textbf{Request} & "
            r"\textbf{Model defaults} \\"
        ),
        r"\midrule",
    ]
    for r in provenance.itertuples():
        if not isinstance(r.digest, str) or not r.digest:
            lines.append(
                f"  {r.model} & -- & think unrecorded & provider default" + r" \\"
            )
            continue
        think = "think unrecorded" if r.think == "unrecorded" else "think off"
        request = think + (", no draft" if getattr(r, "draft", "") == "off" else "")
        same = captured_digest.get(r.tag) == r.digest
        defaults = by_tag.get(r.tag, "") if same else "unrecorded"
        lines.append(
            f"  {r.model} & \\texttt{{{r.digest}}} & {request}"
            f" & \\texttt{{{_tex(defaults) or '--'}}}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def _label_driver(gpu: str) -> str:
    """`nvidia-smi` reports "<name>, <driver>"; name the second field."""

    name, sep, driver = str(gpu).rpartition(", ")
    return f"{name}, NVIDIA driver {driver}" if sep else str(gpu)


# A sweep finished more than this long before the provenance snapshot is not
# covered by the digest the snapshot recorded.
STALE_DIGEST_DAYS = 7


def _days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def provenance_to_latex(table: pd.DataFrame, out_path: Path) -> None:
    """The provenance table, for the appendix."""

    served = table[table["architecture"].astype(bool)]
    # One serving stack for every model: list it once, not once per row.
    host = ", ".join(
        sorted(
            {
                f"Ollama {r.runtime}, {_label_driver(r.gpu)}"
                for r in served.itertuples()
                if r.runtime
            }
        )
    )
    snapshot = str(served["snapshot_taken"].iloc[0]) if len(served) else ""
    # A digest read on the snapshot date says nothing about a sweep run long
    # before it: the tag may have been re-pulled in between.
    stale = [
        r.model
        for r in served.itertuples()
        # A system not collected yet has no window to compare.
        if snapshot
        and isinstance(r.collected_to, str)
        and r.collected_to
        and _days_between(r.collected_to, snapshot) > STALE_DIGEST_DAYS
    ]
    stale_note = (
        f" For {_join(stale)}, collected before that date, the digest does not "
        r"establish which weights were used."
        if stale
        else ""
    )
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Provenance of the served models. Tag is the identifier "
            r"passed to the runtime and Digest the manifest hash the runtime "
            rf"resolved that tag to on {snapshot}, when this table was "
            r"captured; the tag is mutable and the digest identifies the "
            rf"weights.{stale_note} Parameters and quantization are as the "
            r"runtime reports them, not as the model card advertises them. "
            r"Trace is the reasoning-trace setting recorded in the run "
            r"records. Systems served through a hosted API carry none of "
            r"these and are omitted; their identifier is the dated model "
            r"string in Section~\ref{sec:protocol}.}"
        ),
        r"\label{tab:provenance}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrll}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Tag} & \textbf{Digest} "
            r"& \textbf{Architecture} & \textbf{Params} & \textbf{Ctx} "
            r"& \textbf{Quant.} & \textbf{Trace} \\"
        ),
        r"\midrule",
    ]

    def tex(value: object) -> str:
        """Cell text safe for LaTeX. Quantization names carry underscores."""

        return str(value).replace("_", r"\_")

    for r in served.itertuples():
        lines.append(
            f"  {tex(r.model)} & \\texttt{{{tex(r.tag)}}}"
            f" & \\texttt{{{tex(r.digest)}}}"
            f" & \\texttt{{{tex(r.architecture)}}} & {tex(r.parameters)}"
            f" & {tex(r.context_length)}"
            f" & {tex(r.quantization)} & {tex(r.think)}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}"]
    if host:
        lines.append(rf"\\[2pt]\footnotesize Serving stack: {host}.")
    lines += [r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    tables = profile.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    provenance = pd.read_csv(
        profile.raw_dir / "model_provenance.csv", dtype=str, keep_default_na=False
    )
    provenance_to_latex(provenance, tables / "provenance.tex")
    params = profile.raw_dir / "model_parameters.csv"
    if params.is_file():
        prompts_to_latex(tables / "protocol_prompts.tex")
        decoding_to_latex(
            pd.read_csv(params, dtype=str, keep_default_na=False),
            provenance,
            tables / "protocol_decoding.tex",
        )


if __name__ == "__main__":
    main()
