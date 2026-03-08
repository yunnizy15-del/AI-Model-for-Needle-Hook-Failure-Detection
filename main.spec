# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = []
datas += collect_data_files("matplotlib")
datas += [("logo.ico", ".")]


a = Analysis(
    ["gui_app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "tkinter",
        "matplotlib.backends.backend_tkagg",
    ]
    + collect_submodules("scipy.stats")
    + collect_submodules("scipy.special")
    + collect_submodules("sklearn"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    module_collection_mode={"scipy": "py", "sklearn": "pyz"},
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="needle_hook_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="logo.ico",
)
