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
        'ghv4',
        'ghv4.config',
        'ghv4.csi_parser',
        'ghv4.serial_io',
        'ghv4.cell_logic',
        'ghv4.spacing_estimator',
        'ghv4.eda_utils',
        'ghv4.viz',
        'ghv4.inference',
        'ghv4.preprocess',
        'ghv4.train',
        'ghv4.ui',
        'ghv4.ui.app',
        'ghv4.ui.capture_tab',
        'ghv4.ui.debug_tab',
        'ghv4.ui.spacing_tab',
        'ghv4.ui.widgets',
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
    name='GHV4_Collector',
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
