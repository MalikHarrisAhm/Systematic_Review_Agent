#!/Users/malikahmed/Documents/DosesAI/Networking/Systematic-review-agent-main/.venv/bin/python3.11
# -*- coding: utf-8 -*-
import re
import sys
from biorun.scripts.fasta_filter import run
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run())
