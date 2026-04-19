import json, os
for root, dirs, files in os.walk('Week07'):
    for f in sorted(files):
        if f.endswith('.ipynb') and 'Day' in f:
            with open(os.path.join(root, f), 'r') as fp:
                nb = json.load(fp)
            imgs = []
            for i, c in enumerate(nb['cells']):
                src = ''.join(c['source'])
                if c['cell_type'] == 'markdown' and 'data:image/png' in src:
                    imgs.append((i, len(src)//1024))
            print(f'{f}: {len(imgs)} images at cells {[i for i,_ in imgs]} ({[s for _,s in imgs]}KB)')
for root, dirs, files in os.walk('Week08'):
    for f in sorted(files):
        if f.endswith('.ipynb') and 'Day' in f:
            with open(os.path.join(root, f), 'r') as fp:
                nb = json.load(fp)
            imgs = []
            for i, c in enumerate(nb['cells']):
                src = ''.join(c['source'])
                if c['cell_type'] == 'markdown' and 'data:image/png' in src:
                    imgs.append((i, len(src)//1024))
            print(f'{f}: {len(imgs)} images at cells {[i for i,_ in imgs]} ({[s for _,s in imgs]}KB)')
