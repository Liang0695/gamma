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


def check_loaded_module_sources(expected, roots):
    """Check real files, or prove the two PyTorch singleton namespaces' origin.

    Never resolve a relative pseudo filename against cwd or exempt a whole
    torch.* prefix. Read dictionaries to avoid triggering dynamic __getattr__.
    """
    def fail(name):
        raise PolicyViolation('runtime_environment_mismatch', 'Shadow imported module: '+name)
    dynamic = {}
    known = {'torch.ops': ('_ops', '_Ops', 'ops'),
             'torch.classes': ('_classes', '_Classes', 'classes')}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in CORE.values(): continue
        if name in known:
            impl_name, class_name, attr = known[name]
            parent = sys.modules.get('torch')
            implementation = sys.modules.get('torch.'+impl_name)
            if parent is None or implementation is None: fail(name)
            parent_values = vars(parent); impl_values = vars(implementation)
            approved_parent = os.path.realpath(expected['modules']['torch']['path'])
            parent_file = parent_values.get('__file__')
            impl_file = impl_values.get('__file__')
            approved_impl = str(Path(approved_parent).parent/(impl_name+'.py'))
            cls = impl_values.get(class_name)
            init = vars(cls).get('__init__') if isinstance(cls, type) else None
            init_code = getattr(init, '__code__', None)
            spec = impl_values.get('__spec__')
            values = vars(module)
            if (not parent_file or not Path(parent_file).is_absolute()
                    or os.path.realpath(parent_file) != approved_parent
                    or parent_values.get('__version__') != expected['modules']['torch']['version']
                    or not any(Path(approved_parent).is_relative_to(Path(base)) for base in roots)
                    or not impl_file or not Path(impl_file).is_absolute()
                    or os.path.realpath(impl_file) != approved_impl
                    or getattr(spec, 'origin', None) != impl_file
                    or type(module) is not cls
                    or cls.__module__ != 'torch.'+impl_name or cls.__name__ != class_name
                    or init_code is None or not Path(init_code.co_filename).is_absolute()
                    or os.path.realpath(init_code.co_filename) != approved_impl
                    or getattr(init, '__globals__', None) is not impl_values
                    or impl_values.get('torch') is not parent
                    or parent_values.get(attr) is not module
                    or impl_values.get(attr) is not module
                    or values.get('__name__') != name or values.get('__spec__') is not None
                    or vars(cls).get('__file__') != impl_name+'.py'
                    or values.get('__file__', impl_name+'.py') != impl_name+'.py'):
                fail(name)
            dynamic[name] = {'parent': approved_parent, 'implementation': approved_impl,
                             'class': 'torch.'+impl_name+'.'+class_name,
                             'singleton_bound': True, 'pseudo_file': impl_name+'.py'}
            continue
        path = getattr(module, '__file__', None)
        if path and (not isinstance(path, str) or not Path(path).is_absolute()
                     or not any(Path(path).resolve().is_relative_to(Path(base)) for base in roots)):
            fail(name)
    return dynamic


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
    check_loaded_module_sources(expected, roots)
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
    dynamic = check_loaded_module_sources(expected, roots)
    return {'executable':executable, 'python_version':version,
            'prefix':os.path.realpath(sys.prefix), 'distributions':found, 'modules':modules,
            'resolved_before_import':resolved_sources, 'dynamic_modules':dynamic}


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
