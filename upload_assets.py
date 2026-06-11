import json
import shutil
import subprocess
import tempfile
import os

manifest = json.load(open("scripts/model_manifest.json"))
tag = manifest["release_tag"]

with tempfile.TemporaryDirectory() as tmp:
    for art in manifest["artifacts"]:
        asset = os.path.join(tmp, art["path"].replace("/", "__"))

        shutil.copy(art["path"], asset)

        subprocess.run(
            ["gh", "release", "upload", tag, asset, "--clobber"],
            check=True
        )

        print("uploaded", art["path"])