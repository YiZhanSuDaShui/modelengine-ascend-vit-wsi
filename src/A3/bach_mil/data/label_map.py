PHOTO_LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
PHOTO_TO_INDEX = {k: i for i, k in enumerate(PHOTO_LABELS)}

# 你可以把 WSI 的 label_names 改成 3 类：['Benign', 'InSitu', 'Invasive']
WSI_LABELS_DEFAULT = ['Normal', 'Benign', 'InSitu', 'Invasive']


def normalize_xml_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace('_', '').replace('-', '').replace(' ', '')
    if 'normal' in s:
        return 'Normal'
    if 'benign' in s:
        return 'Benign'
    if 'insitu' in s or 'in situ' in s:
        return 'InSitu'
    if 'invasive' in s:
        return 'Invasive'
    return None
