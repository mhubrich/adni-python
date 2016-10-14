import xml.etree.ElementTree as ET
import glob
import os

from utils.config import config


class Scan:
    def __init__(self, subject, imageID, gender, age, group, tracer, manufacturer, path):
        self.subject = subject
        self.imageID = imageID
        self.gender = gender
        self.age = age
        self.group = group
        self.tracer = tracer
        self.manufacturer = manufacturer
        self.path = path


def _build_path(base, subject, preprocessing, date, imageID):
    preprocessing = preprocessing.replace(' ', '_')
    date = date.replace(' ', '_')
    date = date.replace(':', '_')
    if config['nii']:
        ext = '*.nii*'
    else:
        ext = '*.npy'
    path = os.path.join(base, subject, preprocessing, date, imageID, ext)
    path = glob.glob(path)
    assert len(path) == 1, \
        "There are %d scans in directory: %s" % (len(path), path)
    return path[0]


def _parse_scan_info(base, filename):
    xml = ET.parse(filename)
    root = xml.getroot()
    nodeSubject = root.find('project').find('subject')
    nodeStudy = nodeSubject.find('study')
    nodeSeries = nodeStudy.find('series')
    nodeSeriesLevelMeta = nodeSeries.find('seriesLevelMeta')
    nodeDerivedProduct = nodeSeriesLevelMeta.find('derivedProduct')
    nodeProtocolTerm = nodeSeriesLevelMeta.find('relatedImageDetail').find('originalRelatedImage').find('protocolTerm')
    subject = nodeSubject.find('subjectIdentifier').text
    assert subject is not None, \
        "Could not find subject in: %s" % filename
    gender = nodeSubject.find('subjectSex').text
    assert gender is not None, \
        "Could not find gender in: %s" % filename
    age = nodeStudy.find('subjectAge').text
    assert age is not None, \
        "Could not find age in: %s" % filename
    for subjectInfo in nodeSubject.findall('subjectInfo'):
        if subjectInfo.attrib['item'] == 'DX Group':
            group = subjectInfo.text
            break
    assert group is not None, \
        "Could not find group in: %s" % filename
    manufacturer = None
    tracer = None
    for protocol in nodeProtocolTerm.findall('protocol'):
        if protocol.attrib['term'] == 'Manufacturer':
            manufacturer = protocol.text
        if protocol.attrib['term'] == 'Radiopharmaceutical':
            tracer = protocol.text
        if manufacturer is not None and tracer is not None:
            break
    assert manufacturer is not None, \
        "Could not find manufacturer in: %s" % filename
    assert tracer is not None, \
        "Could not find tracer in: %s" % filename
    imageID = nodeDerivedProduct.find('imageUID').text
    assert imageID is not None, \
        "Could not find image ID in: %s" % filename
    imageID = 'I' + imageID
    preprocessing = nodeDerivedProduct.find('processedDataLabel').text
    date = nodeSeries.find('dateAcquired').text
    path = _build_path(base, subject, preprocessing, date, imageID)
    return Scan(subject, imageID, gender, age, group, tracer, manufacturer, path)


def load_scans(directory):
    files = glob.glob(os.path.join(directory, '*.xml'))
    scans = []
    for f in files:
        scans.append(_parse_scan_info(directory, f))
    return scans
