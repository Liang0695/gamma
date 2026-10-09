"""Real installed CPU PyTorch namespace provenance; no model or GPU loading."""
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from tests._tmp import temp_dir_outside_repo
from v3.common.errors import PolicyViolation
from v3.train.runtime_environment import check_loaded_module_sources


class DynamicTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.expected = {'modules':{'torch':{'path':torch.__file__,'version':torch.__version__}}}
        cls.roots = {str(Path(torch.__file__).resolve().parents[1])}

    def check(self):
        return check_loaded_module_sources(self.expected,self.roots)

    def test_real_singletons_have_same_proof_in_all_working_directories(self):
        before=Path.cwd()
        with temp_dir_outside_repo('dynamic_cwd_') as scratch:
            directories=[before,Path(scratch),Path(next(iter(self.roots)))]
            results=[]
            try:
                for directory in directories:
                    os.chdir(directory)
                    results.append(self.check())
            finally:
                os.chdir(before)
        self.assertEqual(set(results[0]),{'torch.ops','torch.classes'})
        self.assertTrue(all(result==results[0] for result in results))

    def test_name_and_relative_file_do_not_authorize_a_fake_object(self):
        for name in ('torch.ops','torch.classes'):
            module=types.ModuleType(name);module.__file__='_'+name.split('.')[1]+'.py'
            with self.subTest(name=name), patch.dict(sys.modules,{name:module}):
                with self.assertRaises(PolicyViolation):self.check()

    def test_actual_class_instance_must_be_the_implementation_singleton(self):
        for attr,impl_name,class_name in (('ops','_ops','_Ops'),('classes','_classes','_Classes')):
            impl=vars(self.torch)[impl_name]
            new=vars(impl)[class_name]()
            with self.subTest(name=attr), patch.object(self.torch,attr,new), \
                 patch.dict(sys.modules,{'torch.'+attr:new}):
                with self.assertRaises(PolicyViolation):self.check()

    def test_implementation_source_and_parent_source_are_checked(self):
        with temp_dir_outside_repo('dynamic_shadow_') as scratch:
            for module in (self.torch,self.torch._ops,self.torch._classes):
                with self.subTest(module=module.__name__),patch.object(module,'__file__',str(Path(scratch)/'shadow.py')):
                    with self.assertRaises(PolicyViolation):self.check()

    def test_other_relative_or_external_modules_still_refused(self):
        with temp_dir_outside_repo('dynamic_other_') as scratch:
            for file in ('_ops.py',str(Path(scratch)/'shadow.py')):
                module=types.ModuleType('torch.not_an_approved_namespace');module.__file__=file
                previous=Path.cwd()
                try:
                    # Even the cwd that made r8 pass cannot legitimize it.
                    os.chdir(next(iter(self.roots)))
                    with patch.dict(sys.modules,{module.__name__:module}):
                        with self.assertRaises(PolicyViolation):self.check()
                finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
