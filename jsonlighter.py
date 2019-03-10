import json
import itertools
from pathlib import Path
import heapq


in_dir = Path("Data/Results")

for f in in_dir.iterdir():
    if f.is_dir():
        continue
    json_data = open(f).read()
    data = json.loads(json_data)
    data['train'] = data.pop('source')

    fi = data['rf_feature_importances']['importance']
    top_ten = heapq.nlargest(10, fi.keys(), key=lambda k: fi[k])
    data['rf_feature_importances']['importance'] = dict((x, fi[x]) for x in top_ten)
    json.dump(data, open(f.parent/"lite"/f.name, 'w'))

