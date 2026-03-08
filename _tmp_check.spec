# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['_tmp_check.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[]
    + collect_submodules('scipy.stats')
    + collect_submodules('scipy.special')
    + collect_submodules('sklearn'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    module_collection_mode={'scipy': 'py', 'sklearn': 'pyz'},
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='tmp_check',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
