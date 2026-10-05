"""Run a complete study pipeline against authorized, unchanged source workbooks."""
from pathlib import Path
import argparse,json,subprocess,sys

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--study',choices=['Displays','IJIE','Age','all'])
 p.add_argument('--data-root',required=True)
 p.add_argument('--output-dir',default='reproduced')
 a=p.parse_args();here=Path(__file__).resolve().parent
 config=here.parent/'package.json'
 packaged_study=json.loads(config.read_text())['study'] if config.exists() else None
 study=a.study or packaged_study or 'all'
 if packaged_study and study!=packaged_study:
  p.error(f'This archive contains the {packaged_study} pipeline. Omit --study or use --study {packaged_study}.')
 out=Path(a.output_dir).resolve();raw=Path(a.data_root).resolve()
 from mot_common import provenance
 provenance(raw,out/'source_check')
 scripts={'Displays':['reproduce_displays_complete.py'],'IJIE':['01_reproduce_primary.py','02_reproduce_legacy.py','reproduce_ijie_extensions.py'],'Age':['reproduce_age_complete.py']}
 for s,files in scripts.items():
  if study not in ['all',s]:continue
  for f in files:
   subprocess.run([sys.executable,str(here/f),'--data-root',str(raw),'--out',str(out/'results'/s)],check=True)
  subprocess.run([sys.executable,str(here/'make_figures.py'),'--study',s,'--results-root',str(out/'results'),'--figures-root',str(out/'figures')],check=True)
 print('Complete. Individual plotting intermediates under _private are not public-release files.')

if __name__=='__main__':main()
