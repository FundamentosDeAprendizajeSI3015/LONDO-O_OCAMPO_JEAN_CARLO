from pathlib import Path

def save_figure(fig, output_dir: Path, filename: str, dpi: int = 150) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path