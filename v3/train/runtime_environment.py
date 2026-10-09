"""Measure the interpreter and installed 60-wheel closure, without pip/network."""
import importlib
from importlib.machinery import PathFinder
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

from ..common.errors import PolicyViolation

CORE = {'torch':'torch', 'peft':'peft', 'transformers':'transformers',
        'accelerate':'accelerate', 'safetensors':'safetensors',
        'compressed-tensors':'compressed_tensors'}


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def probe_environment(expected, manifest_path):
    def fail(reason):
        raise PolicyViolation('runtime_environment_mismatch', reason)
    executable = os.path.realpath(sys.executable)
    version = platform.python_version()
    if executable != os.path.realpath(expected['executable']) or version != expected['python_version']:
        fail('Wrong interpreter: '+executable+' '+version)
    if os.path.realpath(sys.prefix) != os.path.realpath(expected['prefix']):
        fail('Wrong interpreter prefix')
    roots = {str(Path(p).resolve()) for p in expected['site_packages']}
    if not roots or any(not Path(p).is_relative_to(Path(expected['prefix']).resolve()) for p in roots):
        fail('Distribution roots are outside approved interpreter prefix')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    wanted = {canonical(row['name']): row['version'] for row in manifest['entries']}
    if len(wanted) != 60: fail('Expected the reviewed 60-wheel closure')
    found = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata.get('Name',''))
        if name not in wanted: continue
        if name in found: fail('Duplicate installed distribution: '+name)
        location = str(Path(dist.locate_file('')).resolve())
        if dist.version != wanted[name] or location not in roots:
            fail('Installed distribution version/path drift: '+name+' '+dist.version+' '+location)
        files = list(dist.files or [])
        meta = [f for f in files if str(f).replace('\\','/').endswith('.dist-info/METADATA')]
        if len(meta) != 1: fail('Missing/ambiguous distribution metadata: '+name)
        metadata_path = Path(dist.locate_file(meta[0])).resolve()
        if not metadata_path.is_relative_to(Path(location)):
            fail('Distribution metadata escaped site-packages: '+name)
        found[name] = {'version': dist.version, 'location': location, 'metadata_path': str(metadata_path)}
    if set(found) != set(wanted): fail('Missing distributions: '+str(sorted(set(wanted)-set(found))))
    # Resolve ALL core origins before executing ANY core module: torch can
    # transitively import another core package during its own initialization.
    resolved_sources = {}
    for name in CORE.values():
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, '__file__', None)
        else:
            spec = PathFinder.find_spec(name, sys.path)
            origin = spec.origin if spec is not None else None
        if not origin:
            fail('Unresolvable core module origin: '+name)
        actual = str(Path(origin).resolve())
        if (actual != os.path.realpath(expected['modules'][name]['path']) or
                not any(Path(actual).is_relative_to(Path(base)) for base in roots)):
            fail('Core module origin rejected BEFORE import: '+name+' '+actual)
        resolved_sources[name] = actual
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in CORE.values(): continue
        path = getattr(module, '__file__', None)
        if path and not any(Path(path).resolve().is_relative_to(Path(base)) for base in roots):
            fail('Preloaded shadow module rejected BEFORE import: '+name)
    modules = {}
    for distribution, name in CORE.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            fail('Cannot import '+name+': '+str(exc))
        path = getattr(module,'__file__',None)
        actual = str(Path(path).resolve()) if path else None
        module_version = getattr(module,'__version__',None)
        if str(module_version).split('+',1)[0] != wanted[distribution]:
            fail('Imported module disagrees with locked release: '+name)
        approved = expected['modules'][name]
        if actual != os.path.realpath(approved['path']) or module_version != approved['version']:
            fail('Imported module version/path drift: '+name)
        if not any(Path(actual).is_relative_to(Path(base)) for base in roots):
            fail('Imported module outside approved environment: '+name)
        modules[name] = {'path': actual, 'version': module_version}
    # Catch preloaded shadow submodules as well as the six direct imports.
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in CORE.values(): continue
        path = getattr(module,'__file__',None)
        if path and not any(Path(path).resolve().is_relative_to(Path(base)) for base in roots):
            fail('Shadow imported module: '+name)
    return {'executable':executable, 'python_version':version,
            'prefix':os.path.realpath(sys.prefix), 'distributions':found, 'modules':modules,
            'resolved_before_import':resolved_sources}


def verify_deployed_environment():
    from .deployment import read_publication
    installed = read_publication()
    expected = installed['approval']['environment']
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root/'v3/locks/train.lock.json').read_text(encoding='utf-8'))
    if expected['python_version'] != lock['python']:
        raise PolicyViolation('runtime_environment_mismatch', 'Approved Python conflicts with lock')
    manifest_path = root / lock['train_stack_manifest']['path']
    return probe_environment(expected, manifest_path)
