def sort_groups(scans):
    groups = {}
    names = []
    for scan in scans:
        if scan.group not in names:
            names.append(scan.group)
            groups[scan.group] = []
        groups[scan.group].append(scan)
    return groups, names


def sort_manufacturer(scans):
    manu = {}
    names = []
    for scan in scans:
        if scan.manufacturer not in names:
            names.append(scan.manufacturer)
            manu[scan.manufacturer] = []
        manu[scan.manufacturer].append(scan)
    return manu, names


def sort_subjects(scans):
    subjects = {}
    names = []
    for scan in scans:
        if scan.subject not in names:
            names.append(scan.subject)
            subjects[scan.subject] = []
        subjects[scan.subject].append(scan)
    return subjects, names
