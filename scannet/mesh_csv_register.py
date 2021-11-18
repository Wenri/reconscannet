import csv
import os
import pathlib
import re
from collections import namedtuple

RegisterFieldsA = namedtuple('RegisterFieldsA', ('Bad', 'Good', 'Fine', 'Fail'))
RegisterFieldsB = namedtuple('RegisterFieldsB', ('Good', 'Fail'))
RegisterSummary = namedtuple('RegisterSummary', ('PerfectA', 'PerfectB', 'Trusted', 'Usable'), defaults=(False,) * 4)


class MeshRegister:
    CSV_TITLE_1 = ('标注方式1', '标注方式2')
    CSV_TITLE_2 = ('很差（目测60%以上都要涂黑）：', '很好（轮廓完整，且无冗余，不需要涂黑）',
                   '部分涂黑之后，可以达到“很好”', '重建失败（文件打开后没有点）',
                   '很好（不需要涂红）', '重建失败（文件打开后没有点）')
    CSV_TITLE_2A = ('很差（目测60%以上都要涂黑）：', '很好（轮廓完整，且无冗余，不需要涂黑）',
                    '部分涂黑之后，可以达到“很好”', '空/文件打不开')
    CSV_TITLE_2B = ('很好（不需要涂红）', '文件打不开/为空')
    SPLITTER = re.compile(r'[^a-zA-Z0-9]+')

    def __init__(self, args):
        self.registerA = {}
        self.registerB = {}
        if args.csv_file.exists():
            self.load_registerAB(args.csv_file)

        csv_black = args.csv_file.with_suffix('.black.csv')
        if csv_black.exists():
            self.load_registerA(csv_black)

        csv_red = args.csv_file.with_suffix('.red.csv')
        if csv_red.exists():
            self.load_registerB(csv_red)
        self.start_from = args.start_from

    @classmethod
    def split_scan_name(cls, scan_name):
        scan_name = pathlib.PureWindowsPath(scan_name.strip())
        scan_name = os.path.join(scan_name.parent.name, scan_name.name)
        if scan_name:
            sp = [s for s in cls.SPLITTER.split(scan_name) if s.isnumeric()]
            return 'scan' + sp[0].strip(), sp[1].strip()
        else:
            return None

    def set_to_registerA(self, scan_name, fieldId):
        scan_info = self.registerA.get(scan_name)
        if scan_info is None:
            scan_info = (False,) * len(RegisterFieldsA._fields)
        scan_info = list(scan_info)
        scan_info[fieldId] = True
        self.registerA[scan_name] = RegisterFieldsA(*scan_info)

    def set_to_registerB(self, scan_name, fieldId):
        scan_info = self.registerB.get(scan_name)
        if scan_info is None:
            scan_info = (False,) * len(RegisterFieldsB._fields)
        scan_info = list(scan_info)
        scan_info[fieldId] = True
        self.registerB[scan_name] = RegisterFieldsB(*scan_info)

    def load_registerA(self, file_name):
        header_seen = 0

        with file_name.open() as f:
            for row in csv.reader(f):
                if not header_seen:
                    assert row[0] == self.CSV_TITLE_1[0]
                    header_seen += 1
                    continue
                elif header_seen == 1:
                    assert tuple(row) == self.CSV_TITLE_2A
                    header_seen += 1
                    continue
                for i, s in enumerate(row):
                    scan_name = self.split_scan_name(s)
                    self.set_to_registerA(scan_name, i)

    def load_registerB(self, file_name):
        header_seen = 0

        with file_name.open() as f:
            for row in csv.reader(f):
                if not header_seen:
                    assert row[0] == self.CSV_TITLE_1[1]
                    header_seen += 1
                    continue
                elif header_seen == 1:
                    assert tuple(row) == self.CSV_TITLE_2B
                    header_seen += 1
                    continue
                for i, s in enumerate(row):
                    scan_name = self.split_scan_name(s)
                    self.set_to_registerB(scan_name, i)

    def load_registerAB(self, file_name):
        filesA_len = len(RegisterFieldsA._fields)
        header_seen = 0

        with file_name.open() as f:
            for row in csv.reader(f):
                if not header_seen:
                    assert row[1] == self.CSV_TITLE_1[0] and row[filesA_len + 1] == self.CSV_TITLE_1[1]
                    header_seen += 1
                    continue
                elif header_seen == 1:
                    assert tuple(row[1:]) == self.CSV_TITLE_2
                    header_seen += 1
                    continue
                scan_name = tuple(row[0].split('_')[:2]) if row[0] else None
                assert scan_name not in self.registerA and scan_name not in self.registerB
                self.registerA[scan_name] = RegisterFieldsA._make(bool(a.strip()) for a in row[1:filesA_len + 1])
                self.registerB[scan_name] = RegisterFieldsB._make(bool(a.strip()) for a in row[filesA_len + 1:])

    def check_scan(self, scan_name, instance_id):
        scan_key = (scan_name.replace('_', ''), instance_id.split('_')[0])
        a = self.registerA.get(scan_key)
        if a is None:
            a = RegisterFieldsA(False, False, False, False)
        b = self.registerB.get(scan_key)
        if b is None:
            b = RegisterFieldsB(False, False)
        if a.Bad or a.Fail or b.Fail or int(scan_key[0][4:]) < self.start_from:
            return RegisterSummary()
        return RegisterSummary(Usable=True, PerfectA=a.Good, PerfectB=b.Good, Trusted=a.Good or a.Fine or b.Good)
