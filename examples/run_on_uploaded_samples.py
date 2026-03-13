from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.pipeline import ExtractionPipeline


if __name__ == '__main__':
    sample_inputs = [
        Path('/mnt/data/E0080082 01-06_BOM.pdf'),
        Path('/mnt/data/E0047678 01-01_BOM.pdf'),
        Path('/mnt/data/E0181296 01-06_BOM.pdf'),
    ]
    for pdf in sample_inputs:
        if not pdf.exists():
            continue
        out_dir = Path('output_sample') / pdf.stem.replace(' ', '_')
        pipeline = ExtractionPipeline(
            ExtractionConfig(
                output_dir=out_dir,
                write_csv=True,
                write_parquet=False,
            )
        )
        rows = pipeline.parse_input(pdf)
        print(pdf.name, len(rows))
