"""Download the attributed UCI SMS source only; never executes archive contents."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
URL='https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    with urllib.request.urlopen(URL,timeout=60) as response:raw=response.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('Unexpected archive size')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info=archive.getinfo('SMSSpamCollection')
        if info.file_size>2_000_000:raise ValueError('Unexpected member size')
        member=archive.read(info)
    a.output.mkdir(parents=True);(a.output/'source.zip').write_bytes(raw);(a.output/'SMSSpamCollection').write_bytes(member)
    metadata={'url':URL,'dataset_page':'https://archive.ics.uci.edu/dataset/228/sms+spam+collection','license':'CC-BY-4.0','citation':'Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.','archive_sha256':hashlib.sha256(raw).hexdigest(),'member_sha256':hashlib.sha256(member).hexdigest()}
    (a.output/'source.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
