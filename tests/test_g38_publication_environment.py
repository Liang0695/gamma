"""CPU fixtures: publication files are not production approvals or GPU evidence."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types
import unittest
from unittest.mock import patch

from tests._tmp import temp_dir_outside_repo
from v3.train import deployment, runtime_environment as env, v6_approval, runner
from v3.common.errors import PolicyViolation
from v3.train import supervised_check, engineering_check, input_binding
from dataclasses import asdict
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'docs/v3/evidence/kaggle-38-train-stack-manifest.json'
ENTRIES = json.loads(MANIFEST.read_text(encoding='utf-8'))['entries']


class PublicationTests(unittest.TestCase):
    def make_publication(self, root):
        root = Path(root)
        evidence = b'CPU publication fixture only'
        approval = {'not_after_epoch':time.time()+300,
                    'evidence_sha256':hashlib.sha256(evidence).hexdigest()}
        (root/'evidence.json').write_bytes(evidence)
        (root/'approval.json').write_text(json.dumps(approval),encoding='utf-8')
        (root/'launch.py').write_bytes(b'# CPU fixture\n')
        pub = {'repo_root':str(ROOT), 'source':{'issue_id':deployment.ISSUE_ID,
            'publisher_agent_id':deployment.PUBLISHER_ID, 'operator_agent_id':deployment.OPERATOR_ID,
            'comment_id':'00000000-0000-0000-0000-000000000001','retrieval':'multica-authenticated-cli'},
            'sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ('approval.json','evidence.json','launch.py')}}
        (root/'publication.json').write_text(json.dumps(pub),encoding='utf-8')
        return pub

    def test_fixed_external_publication_and_tampering(self):
        with temp_dir_outside_repo('publication_') as path:
            root = Path(path).resolve()
            pub = self.make_publication(root)
            # Only filesystem mode probing is injected on Windows; real file
            # reads, source validation and digest checks execute unmodified.
            with patch.object(deployment,'DEPLOYMENT_ROOT',root), \
                 patch.object(deployment,'_permissions') as perms:
                result = deployment.read_publication()
                self.assertEqual(result['anchor'],pub['sha256']['approval.json'])
                self.assertGreaterEqual(perms.call_count,5)
                self.assertEqual(v6_approval.deployment_approval_anchor(),result['anchor'])
                for name in ('approval.json','evidence.json','launch.py'):
                    original = (root/name).read_bytes()
                    (root/name).write_bytes(original+b' ')
                    with self.subTest(member=name), self.assertRaises(PolicyViolation):
                        deployment.read_publication()
                    (root/name).write_bytes(original)
                pub['source']['publisher_agent_id'] = 'self'
                (root/'publication.json').write_text(json.dumps(pub))
                with self.assertRaises(PolicyViolation): deployment.read_publication()

    def test_working_tree_and_environment_override_cannot_publish(self):
        with patch.object(deployment,'DEPLOYMENT_ROOT',ROOT), \
             patch.dict(os.environ,{'V3_DEPLOYMENT_ROOT':str(ROOT)}):
            with self.assertRaises(PolicyViolation): deployment.read_publication()
        self.assertNotEqual(str(deployment.DEPLOYMENT_ROOT),str(ROOT))

    def test_posix_owner_modes_and_symlink_controls(self):
        # Exercise target checks using stat fixtures; not a Windows ACL claim.
        fake_os = types.SimpleNamespace(name='posix',getuid=lambda:1000)
        with patch.object(deployment,'os',fake_os):
            for mode,uid,ok in ((stat.S_IFREG|0o400,1000,True),
                                (stat.S_IFREG|0o600,1000,False),
                                (stat.S_IFREG|0o444,999,False),
                                (stat.S_IFLNK|0o777,1000,False)):
                p=types.SimpleNamespace(lstat=lambda:types.SimpleNamespace(st_mode=mode,st_uid=uid))
                if ok: deployment._permissions(p,immutable=True)
                else:
                    with self.assertRaises(PolicyViolation): deployment._permissions(p,immutable=True)


class FakeDistribution:
    def __init__(self,row,site):
        self.metadata={'Name':row['name']};self.version=row['version'];self.site=site
        self.files=[Path(row['name'].replace('-','_')+'-'+self.version+'.dist-info/METADATA')]
    def locate_file(self,item):return self.site/item


class EnvironmentTests(unittest.TestCase):
    def test_every_core_shadow_rejects_before_any_module_side_effect(self):
        for target in env.CORE.values():
            with self.subTest(target=target), temp_dir_outside_repo('import_sentinel_') as path:
                root=Path(path).resolve()
                expected,distributions,_=self.fixture(root/'approved')
                site=Path(expected['site_packages'][0])
                shadow=root/'shadow';shadow.mkdir()
                sentinel=root/'executed.txt'
                for name,record in expected['modules'].items():
                    source=Path(record['path']);source.parent.mkdir()
                    source.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("executed")\n__version__='+repr(record['version'])+'\n',encoding='utf-8')
                shadow_file=shadow/(target+'.py')
                shadow_file.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("SHADOW")\n',encoding='utf-8')
                with patch.dict(sys.modules), patch.object(sys,'path',[str(shadow),str(site),*sys.path]), \
                     patch.object(env.metadata,'distributions',return_value=distributions), \
                     patch.object(env.platform,'python_version',return_value='3.11.9'), \
                     patch.object(env.sys,'prefix',expected['prefix']):
                    for name in list(sys.modules):
                        if name.split('.')[0] in env.CORE.values():del sys.modules[name]
                    with self.assertRaisesRegex(PolicyViolation,'BEFORE import'):
                        env.probe_environment(expected,MANIFEST)
                    self.assertFalse(sentinel.exists())
                    # Positive control: exactly the approved modules can execute.
                    shadow_file.unlink()
                    env.importlib.invalidate_caches()
                    measured=env.probe_environment(expected,MANIFEST)
                    self.assertTrue(sentinel.exists())
                    self.assertEqual(len(measured['resolved_before_import']),6)

    def test_isolated_bootstrap_ignores_pythonpath_home_and_user_site(self):
        with temp_dir_outside_repo('isolation_') as path:
            root=Path(path).resolve();shadow=root/'shadow';shadow.mkdir()
            sentinel=root/'executed.txt'
            (shadow/'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("executed")\n')
            poisoned=dict(os.environ,PYTHONPATH=str(shadow),PYTHONHOME=str(root/'wrong'),
                          PYTHONUSERBASE=str(shadow),PYTHONNOUSERSITE='0')
            clean=supervised_check.sanitize_child_env(poisoned)['env']
            self.assertNotIn('PYTHONPATH',clean)
            self.assertNotIn('PYTHONHOME',clean)
            self.assertNotIn('PYTHONUSERBASE',clean)
            self.assertEqual(clean['PYTHONNOUSERSITE'],'1')
            command=supervised_check.isolated_module_command('v3.train.engineering_check')+['--help']
            result=subprocess.run(command,cwd=root,env=poisoned,capture_output=True,
                                  text=True,encoding='utf-8',errors='strict')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--dest-dir',result.stdout)
            self.assertFalse(sentinel.exists())
            report=supervised_check.run_supervised_check(v6_dir=str(ROOT/'tools/v6'),
                dest_dir=str(root/'out'),run_root=str(root/'runs'),steps=2,
                checkpoint_every=1,stop_at_step=1,approved_budget_seconds=60,env=poisoned)
            self.assertEqual(report['overall_exit_code'],0,report)
            self.assertFalse(sentinel.exists())
            self.assertIn('-I',report['supervisor_command'])
            plan=json.loads(Path(report['plan_path']).read_text(encoding='utf-8'))
            for phase in plan['phases']:
                self.assertIn('-I',phase['command'])
    def fixture(self,root):
        prefix=Path(root).resolve();site=prefix/'lib/site-packages';site.mkdir(parents=True)
        distributions=[FakeDistribution(row,site) for row in ENTRIES]
        versions={env.canonical(row['name']):row['version'] for row in ENTRIES}
        modules={}
        for dist,name in env.CORE.items():
            module=types.ModuleType(name)
            module.__version__=versions[dist]+('+cu128' if name=='torch' else '')
            module.__file__=str(site/name/'__init__.py')
            modules[name]=module
        modules['peft'].LoraConfig=object;modules['peft'].get_peft_model=lambda *a:None
        modules['transformers'].AutoModelForCausalLM=object
        modules['transformers'].AutoTokenizer=object
        expected={'executable':sys.executable,'python_version':'3.11.9','prefix':str(prefix),
            'site_packages':[str(site)],'modules':{n:{'path':m.__file__,'version':m.__version__} for n,m in modules.items()}}
        return expected,distributions,modules

    def test_correct_60_package_environment_and_drift(self):
        with temp_dir_outside_repo('env_probe_') as root:
            expected,distributions,modules=self.fixture(root)
            with ExitStack() as stack:
                stack.enter_context(patch.object(env.metadata,'distributions',return_value=distributions))
                stack.enter_context(patch.object(env.platform,'python_version',return_value='3.11.9'))
                stack.enter_context(patch.object(env.sys,'prefix',expected['prefix']))
                stack.enter_context(patch.dict(sys.modules,modules))
                # Existing real torch submodules belong to the local test Python,
                # so isolate the fixture import view without altering production.
                live=dict(sys.modules)
                for name in list(sys.modules):
                    if name.split('.')[0] in env.CORE.values() and '.' in name:
                        del sys.modules[name]
                stack.callback(sys.modules.update,live)
                good=env.probe_environment(expected,MANIFEST)
                self.assertEqual(len(good['distributions']),60)
                self.assertEqual(good['modules']['torch']['version'],'2.10.0+cu128')
                for version in ('0.0.0','999.999.999'):
                    for name in ('torch','peft','transformers'):
                        module=modules[name];original=module.__version__;module.__version__=version
                        with self.subTest(module=name,version=version),self.assertRaises(PolicyViolation):
                            env.probe_environment(expected,MANIFEST)
                        module.__version__=original
                old=distributions[-1].version;distributions[-1].version='999.999.999'
                with self.assertRaises(PolicyViolation):env.probe_environment(expected,MANIFEST)
                distributions[-1].version=old
                for field,bad in (('executable','/wrong/python'),('python_version','0.0.0'),('prefix','/wrong')):
                    changed=copy.deepcopy(expected);changed[field]=bad
                    with self.subTest(field=field),self.assertRaises(PolicyViolation):
                        env.probe_environment(changed,MANIFEST)
                modules['torch'].__file__=str(Path(root)/'shadow.py')
                with self.assertRaises(PolicyViolation):env.probe_environment(expected,MANIFEST)

    def test_import_stack_rechecks_before_returning_modules(self):
        for version in ('0.0.0','999.999.999'):
            fake=types.ModuleType('torch');fake.__version__=version
            with patch.dict(sys.modules,{'torch':fake}), \
                 patch.object(env,'verify_deployed_environment',side_effect=PolicyViolation('runtime_environment_mismatch','fixture drift')) as check:
                with self.assertRaises(PolicyViolation):runner.TorchPeftBackend._import_stack()
                check.assert_called_once()

    def test_published_parent_and_child_reach_model_boundary_and_reject_drift(self):
        with temp_dir_outside_repo('published_flow_') as path:
            root=Path(path).resolve(); deploy=root/'publication';deploy.mkdir()
            PublicationTests().make_publication(deploy)
            plan=engineering_check._plan(2)
            planfile=root/'plan.json';planfile.write_text(json.dumps(asdict(plan)))
            snapshot=root/'snapshot'
            source=ROOT/'docs/v3/evidence/kaggle-38-g9-evidence'
            for name,relative in input_binding.DEFAULT_SNAPSHOT_LAYOUT.items():
                dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source/name,dest)
            bound=input_binding.bind_model_inputs(root=str(snapshot),interface_path=str(ROOT/'v3/locks/official-interface.json'))
            options=v6_approval.runtime_options(steps=2,checkpoint_every=1,stop_at_step=1,
                budget_seconds=30,requested_gpu_hours=0.01,model_id=bound['repo_id'],
                model_revision=bound['revision'],target_modules=['q_proj'])
            expected,distributions,modules=self.fixture(root/'env')
            with ExitStack() as stack:
                stack.enter_context(patch.object(deployment,'DEPLOYMENT_ROOT',deploy))
                stack.enter_context(patch.object(deployment,'_permissions'))
                # Explicit CPU fixtures for platform/installed packages; real
                # publication IO, runtime binding and both entry gates execute.
                stack.enter_context(patch.object(env.metadata,'distributions',return_value=distributions))
                stack.enter_context(patch.object(env.platform,'python_version',return_value='3.11.9'))
                stack.enter_context(patch.object(env.sys,'prefix',expected['prefix']))
                stack.enter_context(patch.dict(sys.modules,modules))
                live=dict(sys.modules)
                for name in list(sys.modules):
                    if name.split('.')[0] in env.CORE.values() and '.' in name:del sys.modules[name]
                stack.callback(sys.modules.update,live)
                stack.enter_context(patch.object(v6_approval.subprocess,'check_output',
                    side_effect=lambda cmd,**kw: 'f'*40 if 'rev-parse' in cmd else ''))
                record=json.loads((deploy/'approval.json').read_text())
                record.update(environment=expected,approval_id='CPU-FIXTURE',target_sha='f'*40,
                              gpu_hours_approved=0.01,verified_by='CPU fixture',verified_utc='fixture')
                def publish():
                    (deploy/'approval.json').write_text(json.dumps(record),encoding='utf-8')
                    pub=json.loads((deploy/'publication.json').read_text())
                    pub['sha256']['approval.json']=hashlib.sha256((deploy/'approval.json').read_bytes()).hexdigest()
                    (deploy/'publication.json').write_text(json.dumps(pub),encoding='utf-8')
                    return pub['sha256']['approval.json']
                publish()
                context=v6_approval.measure_runtime_context(plan=plan,input_binding=bound,
                    local_model_dir=str(source),options=options)
                record['runtime_context']=context;anchor=publish()
                self.assertIs(runner.TorchPeftBackend._import_stack()[0],modules['torch'])
                hashes=root/'hashes.json';hashes.write_text(json.dumps(context['hashes']))
                overrides=dict(backend='torch-peft',plan_json=str(planfile),hashes=str(hashes),
                    model_inputs_root=str(snapshot),local_model_dir=str(source),model_id=bound['repo_id'],
                    model_revision=bound['revision'],target_modules='q_proj',local_load_authorized=True,
                    approval=str(deploy/'approval.json'),approval_expected_sha256=anchor,
                    approval_evidence_ref=str(deploy/'evidence.json'),requested_gpu_hours=0.01)
                parent=supervised_check.run_supervised_check(v6_dir=str(ROOT/'tools/v6'),
                    dest_dir=str(root/'out'),run_root=str(root/'runs'),steps=2,checkpoint_every=1,
                    stop_at_step=1,approved_budget_seconds=30,extra_child_args=overrides,dry_run=True)
                cutoff=supervised_check.read_deadline_file(parent['deadline_file']['path'],parent['deadline_file']['file_sha256'])
                child=dict(dest_dir=str(root/'out'),backend='torch-peft',steps=2,checkpoint_every=1,
                    stop_at_step=1,plan_json=str(planfile),model_inputs_root=str(snapshot),
                    local_model_dir=str(source),model_id=bound['repo_id'],model_revision=bound['revision'],
                    target_modules=['q_proj'],hashes=context['hashes'],local_load_authorized=True,
                    approval_path=str(deploy/'approval.json'),approval_expected_sha256=anchor,
                    approval_evidence_ref=str(deploy/'evidence.json'),requested_gpu_hours=0.01,
                    supervised_deadline=cutoff)
                with patch.object(engineering_check,'_make_backend',side_effect=RuntimeError('CPU boundary reached')) as model:
                    with self.assertRaisesRegex(RuntimeError,'CPU boundary reached'):
                        engineering_check.run_engineering_check(**child)
                    model.assert_called_once()
                modules['torch'].__version__='999.999.999'
                with self.assertRaisesRegex(PolicyViolation,'runtime_environment_mismatch'):
                    runner.TorchPeftBackend._import_stack()
                with patch.object(engineering_check,'_make_backend') as model:
                    with self.assertRaisesRegex(PolicyViolation,'runtime_environment_mismatch'):
                        engineering_check.run_engineering_check(**child)
                    model.assert_not_called()
                with self.assertRaisesRegex(PolicyViolation,'runtime_environment_mismatch'):
                    supervised_check.run_supervised_check(v6_dir=str(ROOT/'tools/v6'),
                        dest_dir=str(root/'out'),run_root=str(root/'runs2'),steps=2,checkpoint_every=1,
                        stop_at_step=1,approved_budget_seconds=30,extra_child_args=overrides,dry_run=True)


if __name__=='__main__':unittest.main()
