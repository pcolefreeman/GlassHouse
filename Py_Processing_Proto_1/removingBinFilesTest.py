import subprocess
import os
import glob

binFolder = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3" # replace to work on your computer

# uncomment to work with linux OS
# result = subprocess.run(
#     f'find "{binFolder}" -name "* .bin" -type f -delete',
#     shell=True,
#     capture_output=True,
#     text=True
# )
#
# if result.returncode == 0:
#     print("All .bin files deleted successfully.")
# else:
#     print(f"Error: {result.stderr}")

# Windows OS lines -> comment out if linux
bin_files = glob.glob(os.path.join(binFolder, "*.bin"))

if not bin_files:
    print("No .bin files found.")
else:
    for file in bin_files:
        os.remove(file)
        print(f"Deleted: {file}")
    print(f"\nDone! {len(bin_files)} file(s) deleted.")