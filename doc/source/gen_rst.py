import sys
import pathlib

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IndexError("No file path provided to generate rst file")

    file_path = pathlib.Path(sys.argv[1])

    file_path = file_path.with_suffix(".rst")

    if file_path.exists():
        exists_input = input("Overwrite existing file ([y]/n) ? ")

        if not exists_input.lower() in ["", "y", "yes"]:
            exit(0)

    file_path.parent.mkdir(exist_ok=True)

    if file_path.stem == "index":
        rst_label = ".. _" + str(file_path.parent).replace("/", ".") + ":"
    else:
        rst_label = (
            ".. _"
            + str(file_path.parent).replace("/", ".")
            + "."
            + file_path.stem
            + ":"
        )

    rst_main_title = file_path.stem.replace("_", " ").capitalize()
    len_rst_main_title = len(rst_main_title)

    with open(file_path, "w") as f:
        f.write(rst_label + "\n")
        f.write("\n")
        f.write("=" * len_rst_main_title + "\n")
        f.write(rst_main_title + "\n")
        f.write("=" * len_rst_main_title + "\n")
