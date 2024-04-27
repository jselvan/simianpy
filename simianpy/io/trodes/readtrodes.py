import re

import numpy as np

def read_header(filename):
    with open(filename, "rb") as f:
        if f.readline().decode("ascii").strip() != "<Start settings>":
            raise Exception("Settings format not supported")
        fieldsText = {}
        for line in f:
            line = line.decode("ascii").strip()
            if line == "<End settings>":
                break
            else:
                key, value = line.split(": ", 1)
                fieldsText[key.lower()] = value
        offset = f.tell()
    return fieldsText, offset

# Main function
def readTrodesExtractedDataFile(filename, mmap_mode=None):
    fieldsText, offset = read_header(filename)
    dtype = parseFields(fieldsText["fields"])
    if mmap_mode is not None:
        data = np.memmap(filename, dtype=dtype, offset=offset, mode=mmap_mode)
    else:
        data = np.fromfile(filename, dtype=dtype, offset=offset)
    return fieldsText, data


# Parses last fields parameter (<time uint32><...>) as a single string
# Assumes it is formatted as <name number * type> or <name type>
# Returns: np.dtype
def parseFields(fieldstr):
    # Returns np.dtype from field string
    sep = re.split("\s", re.sub(r"\>\<|\>|\<", " ", fieldstr).strip())
    typearr = []
    # Every two elmts is fieldname followed by datatype
    for i in range(0, sep.__len__(), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = "uint32"
        # Finds if a <num>* is included in datatype
        if "*" in sep[i + 1]:
            temptypes = re.split("\*", sep[i + 1])
            # Results in the correct assignment, whether str is num*dtype or dtype*num
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i + 1]

        if hasattr(np, ftype):
            typearr.append((str(fieldname), ftype, repeats))
        else:
            raise ValueError(ftype + " is not a valid field type.")
    return np.dtype(typearr)
