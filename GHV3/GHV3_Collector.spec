# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('customtkinter')

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'serial.tools.list_ports',
        'ghv3_1',
        'ghv3_1.config',
        'ghv3_1.csi_parser',
        'ghv3_1.serial_io',
        'ghv3_1.cell_logic',
        'ghv3_1.spacing_estimator',
        'ghv3_1.eda_utils',
        'ghv3_1.viz',
        'ghv3_1.inference',
        'ghv3_1.preprocess',
        'ghv3_1.train',
        'ghv3_1.ui',
        'ghv3_1.ui.app',
        'ghv3_1.ui.capture_tab',
        'ghv3_1.ui.debug_tab',
        'ghv3_1.ui.spacing_tab',
        'ghv3_1.ui.widgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GHV3_Collector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
