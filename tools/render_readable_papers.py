"""Render repository-readable manuscripts from the canonical LaTeX draft text."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
TABLE='''
| Task | Model | Strict | Fence | Macro-F1 | JSON |
|---|---|---:|---:|---:|---:|
| Auth | Base | 0/96 | 32/96 | 0.307 | 0/96 |
| Auth | Seed 17 | 70/96 | 70/96 | 0.734 | 96/96 |
| Auth | Seed 23 | 70/96 | 70/96 | 0.720 | 96/96 |
| Auth | Seed 41 | 80/96 | 80/96 | 0.829 | 96/96 |
| Workflow | Base | 0/96 | 53/96 | 0.440 | 0/96 |
| Workflow | Seed 17 | 96/96 | 96/96 | 1.000 | 96/96 |
| Workflow | Seed 23 | 96/96 | 96/96 | 1.000 | 96/96 |
| Workflow | Seed 41 | 96/96 | 96/96 | 1.000 | 96/96 |

Strict parsing and complete-outer-fence removal are distinct measures. Macro-F1 uses fence-permitting classification; JSON denotes strict schema compliance. Each model receives the same 96 cases for its task.
'''


def render(source):
    text=source.read_text()
    title=re.search(r'\\title\{([^\n]+)\}',text).group(1)
    body=text.split('\\begin{abstract}',1)[1].split('\\end{document}',1)[0]
    body='## Abstract\n\n'+body.replace('\\end{abstract}','')
    body=re.sub(r'\\begin\{table\}.*?\\end\{table\}',lambda m:TABLE,body,flags=re.S)
    body=re.sub(r'\\section\{([^}]+)\}',r'## \1\n',body)
    body=body.replace('Table~\\ref{tab:results}','The table below')
    body=body.replace('\\begin{equation}','\n$$\n').replace('\\end{equation}','\n$$\n')
    body=re.sub(r'\\begin\{thebibliography\}\{[^}]+\}','## References\n',body)
    body=body.replace('\\end{thebibliography}','')
    body=re.sub(r'\\bibitem\{([^}]+)\}',r'- [\1]',body)
    body=re.sub(r'\\cite\{([^}]+)\}',r'[\1]',body)
    body=re.sub(r'\\texttt\{([^}]+)\}',r'`\1`',body)
    body=re.sub(r'\\emph\{([^}]+)\}',r'*\1*',body)
    body=re.sub(r'\\url\{([^}]+)\}',r'[source](\1)',body)
    for old,new in [('\\,',' '),('\\_','_'),('\\%','%'),('~',' '),('\\\\',' ')]:
        title=title.replace(old,new);body=body.replace(old,new)
    body=re.sub(r'\n{3,}','\n\n',body).strip()
    output=f'# {title}\n\n**Megern Qaisse · Independent open research · Working draft, October 2026**\n\nCanonical editable source: [{source.name}]({source.name}). Not peer reviewed.\n\n'+body+'\n'
    source.with_suffix('.md').write_text(output)


if __name__=='__main__':
    for name in ['inference-study','specialist-study']:
        render(ROOT/'papers'/(name+'.tex'))
