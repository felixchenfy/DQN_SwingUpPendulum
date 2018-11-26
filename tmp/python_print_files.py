import pathlib

# Sort the checkpoints by modification time.
checkpoint_dir="./"
checkpoints = pathlib.Path(checkpoint_dir).glob("*")
# checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
# latest = str(checkpoints[-1])
for i in range(len(checkpoints)):
    print(checkpoints[i])